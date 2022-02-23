
from torch import nn, cat, linspace, tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
import torch
from torch import Tensor
from torch.nn.utils.parametrize import register_parametrization
import pytorch_lightning as pl
import numpy as np
from random import uniform
from torchaudio.functional import lowpass_biquad

import fbg_swdm.simulation as sim
import fbg_swdm.variables as vars

# ---------------------------- Model Loading Utils --------------------------- #

def param_to_buffer(module, name, param): 
    """Turns a parameter into a buffer"""
    delattr(module, name) # Unregister parameter
    module.register_buffer(name, param)

def params_to_buffers(module):
    """Turns all parameters of a module into buffers."""
    modules = module.modules()
    module = next(modules)
    for name, param in module.named_parameters(recurse=False):
        param_to_buffer(module, name, param)
    for module in modules:
        params_to_buffers(module)

def convert_state_dict(state_dict, name):
    new_state_dict = dict()
    for k, v in state_dict.items():
        if k == 'weights':
            new_state_dict[k]=v
        else:
            k = name + k
            k = k.replace('.conv.', '.body.')
            new_state_dict[k] = v
    return new_state_dict

def load_old_model(model_class, checkpoint_path, add_hparams):

    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    hparams = checkpoint['hyper_parameters']
    reg = hparams.pop('l1_reg')
    hparams['reg']=reg
    hparams = {**hparams, **add_hparams}

    model = model_class(**hparams)

    state_dict = convert_state_dict(state_dict, name='encoder.')
    model.load_state_dict(state_dict)
    return model

# ---------------------------------- Losses ---------------------------------- #
    
@torch.jit.script
def weighted_mse_func(weights, y, y_hat):
    # weighted regression loss
    reg_loss = torch.dot(weights,torch.mean(F.mse_loss(y_hat, y, 
                                                            reduction='none'), 
                                            dim=0))
    return reg_loss

def weighted_mse(weights):
    def func(y, y_hat):
        return weighted_mse_func(weights, y, y_hat)
    return func

# ------------------------------- Regularizers ------------------------------- #
        
@torch.jit.script
def kl_div_func(rho: float, input: Tensor) -> Tensor:
    # kl divergence for bernulli distribution 
    # sparness constraint
    rho_hat = input.mean(0)
    dkl = - rho * torch.log(rho_hat) - (1-rho)*torch.log(1-rho_hat)
    return dkl.mean()

def kl_div(rho):
    def func(input):
        return kl_div_func(rho, input)
    return func

@torch.jit.script
def roughness(input: Tensor) -> Tensor:
    """Measure of roughness"""
    shape = input.size()
    shape[-1] = 1
    zeros = torch.zeros(shape, device=input.device)
    diff  = torch.diff(input, prepend=zeros, append=zeros, dim=-1)
    norm_diff = (diff - diff.mean(dim=0))/diff.std(dim=0, unbiased=True)
    roughness = norm_diff.diff(dim=0)**2
    return roughness.mean()

@torch.jit.script
def spread_func(input: Tensor, sigma: float=1e-2) -> Tensor:
    # input: B, C, W tensor
    length = input.size(-1)
    x = torch.arange(length, device = input.device)
    mean = torch.sum(x*input.detach(), dim=-1, keepdim=True)
    dist_mean = torch.abs(x - mean) # distance to mean
    dist_mean /= length//2 # normalize
    spread = torch.mean(dist_mean*input**2, dim=-1)
    weight = F.softplus(sigma-torch.max(input.detach(), dim=-1).values, beta=5)
    spread = spread*weight[None,...]
    return spread.mean()

def spread(sigma):
    def func(input):
        return spread_func(input, sigma)
    return func

@torch.jit.script
def kurtosis_func(input: Tensor, kappa: float=10) -> Tensor:
    mean_output = torch.mean(input, dim=-1, keepdim=True).detach()
    std_output = torch.std(input, dim=-1, keepdim=True)
    K = torch.mean((((input - mean_output) / std_output) ** 4), dim=-1)
    return -K.mean()

def kurtosis(kappa):
    def func(input):
        return kurtosis_func(input, kappa)
    return func

@torch.jit.script
def gaussianess_func(input: Tensor, nu: float=1e-2) -> Tensor:
    length = input.size(-1)
    x = torch.arange(length, device = input.device)/length
    mean = torch.sum(x*input.detach(), dim=-1, keepdim=True)
    dist_mean = x - mean
    g = 1/(nu*torch.sqrt(2*vars.π))*torch.exp(-dist_mean**2/(2*nu**2))
    g /= g.sum()
    G = F.relu(input-g)
    G = G**2
    return G.mean()

def gaussianess(nu):
    def func(input):
        return gaussianess_func(input, nu)
    return func

@torch.jit.script
def l1_norm(input: Tensor) -> Tensor:
    return torch.norm(input, p=1, dim=-1).mean()

@torch.jit.script
def roughness(input: Tensor) -> Tensor:
    """Measure of roughness"""
    input = input.squeeze()
    input = input.T
    zeros = torch.zeros(1, input.size(1), device=input.device)
    diff  = torch.diff(input, prepend=zeros, append=zeros, dim=0)
    norm_diff = (diff - diff.mean(dim=0))/diff.std(dim=0, unbiased=True)
    roughness = norm_diff.diff(dim=0)**2
    return roughness.mean()

# -------------------------------- Model Utils ------------------------------- #

def conv_mish(in_channels, out_channels, kernel_size=1, dilation=1, groups=1):
    # create & initialize conv layer for mish activation
    conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same',
                     dilation=dilation, groups=groups)
    nn.init.kaiming_uniform_(conv.weight, a=0.0003)
    nn.init.zeros_(conv.bias)
    return conv

def resample(in_channels, out_channels):
    conv = nn.Conv1d(in_channels, out_channels, 1, padding='same', bias=False)
    nn.init.kaiming_uniform_(conv.weight, a=0.0003)
    return conv

def test_kernel(vect):
    receptive_field = 1
    #dilation = 1
    for kernel in vect:
        receptive_field = (kernel-1)*receptive_field+1
        #dilation = dilation*(kernel-1)
    return receptive_field

def get_kernel_sizes(n_layers, target, verbose=False):
    # if target is in nm
    if isinstance(target, float):
        target = int(target*vars.N/(vars.Δ*2)*vars.n)

    # init all kernels in 3 for each layer
    kernel_vect = [3]*n_layers

    #increase first kernel until receptive_field is surpassed
    while test_kernel(kernel_vect) < target:
        kernel_vect[0] = (kernel_vect[0]+1)*2 -1

    receptive_field = test_kernel(kernel_vect)
    if verbose:
        print('kernel_vect: ', kernel_vect)
        print('receptive_field: ', receptive_field)
    
    # if receptive_field surpasses target without increasing kernel
    if kernel_vect[0]==3:
        if verbose:
            print('Too many layers')
        return kernel_vect

    # Decrease first kernel by roughly 2e-1 times and compensate by
    # incresing by 2 the following kernels to get a receptive_field
    # closer to the target
    best_kernel_vect = kernel_vect.copy()
    best_receptive_field = receptive_field

    while True:
        for change_idx in range(1, n_layers):
            kernel_vect[0] = (kernel_vect[0]+1)//2 -1
            kernel_vect[change_idx] += 2
            receptive_field = test_kernel(kernel_vect)
            if verbose:
                print('kernel_vect: ', kernel_vect)
                print('receptive_field: ', receptive_field)
            if kernel_vect[0]<7 or kernel_vect[0]<kernel_vect[1]:
                if verbose:
                    print('kernel_vect: ', best_kernel_vect)
                    print('receptive_field: ', best_receptive_field)
                return best_kernel_vect
            if abs(receptive_field-target)<abs(best_receptive_field-target):
                best_kernel_vect = kernel_vect.copy()
                best_receptive_field = receptive_field


# ------------------------------- Augmentation ------------------------------- #

@torch.jit.script
def batch_shift(batch: Tuple[Tensor, Tensor], n: int) -> Tuple[Tensor, Tensor]:
    """ Extend batch by concatenating n shifts around zero-shift"""
    n += 1 #include zero-shift
    x, y = batch 
    shifts = torch.arange(n, device=x.device)-n//2
    x = torch.cat([torch.roll(x, int(s)) if s else x for s in shifts], dim=0)
    y = y[None, ...] + vars.Δ/vars.N*shifts[..., None, None] #add shift on first dimension
    y = y.reshape(-1, y.size(-1)) # concat new dim along the first dimension
    batch = x, y
    return batch

# ----------------------------- Parametrizations ----------------------------- #
    

class SigmoidStraightThrough(torch.autograd.Function):

    @staticmethod
    def forward(ctx, u):
        return sigmoid(u)

    @staticmethod
    def backward(ctx, dx):
        return dx
def sigmoid_straight_through(u):
    return SigmoidStraightThrough.apply(u)

class Symmetric(nn.Module):
    """Reparametrize a vector to a symmetrical one by concatenating a reflexion"""
    def __init__(self, kernel_size):
        super().__init__()
        self.N = kernel_size//2+1
    
    def forward(self, X):
        return torch.cat((X, X[..., :-1].flip(-1)), dim=-1)
    
    def right_inverse(self, Z):
        return Z[...,:self.N]

class UnitCap(nn.Module):
    """Reparametrize a variable to be bounded in [0, 1] with a sigmoid function 
       using the natural gradient"""
    def forward(self, X):
        return sigmoid_straight_through(X)
    def right_inverse(self, Z):
        return torch.logit(Z, eps=1e-6) #needs bound otherwise inf

class Clamp(nn.Module):
    """Reparametrize a variable to be bounded in [0, 1] with a clamp"""
    def forward(self, X):
        return X.clamp(0, 1)
    def right_inverse(self, Z):
        return Z

class ChannelSame(nn.Module):
    """Reparametrize convolution weight to repeat same kernel for every channel"""
    def __init__(self, Q):
        super().__init__()
        self.Q = Q
    def forward(self, X):
        return X.expand((self.Q, -1, -1))
    def right_inverse(self, Z):
        return Z[:1]

class ChannelSameBias(nn.Module):
    """Reparametrize convolution bias to repeat same kernel for every channel"""
    def __init__(self, Q):
        super().__init__()
        self.Q = Q
    def forward(self, X):
        return X.expand(self.Q)
    def right_inverse(self, Z):
        return Z[:1]


class NormConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, in_channels, out_channels, kernel_size, padding, groups,
                 bias=False):
        super().__init__(in_channels, out_channels, kernel_size, 
                         padding=padding, groups=groups, bias=bias)
        parametrize.register_parametrization(self, "weight", UnitCap())

class SymmetricNormConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, in_channels, out_channels, kernel_size, padding, groups,
                 bias=False):
        super().__init__(in_channels, out_channels, kernel_size, 
                         padding=padding, groups=groups, bias=bias)
        # TODO: replace this for more elegant getattr and setattr in Symmetric
        with torch.no_grad():
            self.weight.data = self.weight.data[..., :kernel_size//2 + 1] # chop kernel dim
        # parametrize.register_parametrization(self, "weight", Clamp())
        parametrize.register_parametrization(self, "weight", UnitCap())
        parametrize.register_parametrization(self, "weight", Symmetric(kernel_size), unsafe=True)
        
class SymmetricConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups,
                 bias=True):
        super().__init__()
        kernel_size = kernel_size//2 + 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                         padding='same', groups=groups, bias=bias)
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='linear')
        parametrize.register_parametrization(self.conv, "weight", Symmetric(), unsafe=True)
    def forward(self, input):
        return self.conv(input)

class NarrowConv1D(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, groups,
                 bias=False):
        super().__init__(in_channels, out_channels, kernel_size, 
                         padding='same', bias=bias, groups=groups)
        with torch.no_grad():
            if bias:
                self.bias.data = self.bias.data[:1]
            self.weight.data = self.weight.data[:1] # chop channel dim
            self.weight.data = self.weight.data[..., :kernel_size//2 + 1] # chop kernel dim
        nn.init.dirac_(self.weight) # Init to identity
        parametrize.register_parametrization(self, "weight", ChannelSame(in_channels), unsafe=True)
        if bias:
            parametrize.register_parametrization(self, "bias", ChannelSameBias(in_channels), unsafe=True)
        parametrize.register_parametrization(self, "weight", Symmetric(kernel_size), unsafe=True)



# ---------------------------------------------------------------------------- #
#                                 Torch Models                                 #
# ---------------------------------------------------------------------------- #

class dense_encoder(nn.Module):
    def __init__(self, num_layers, num_head_layers):
        super().__init__()

        self.num_layers = num_layers
        self.num_head_layers = num_head_layers
        
        #Conv layers with concat connections
        self.body = nn.ModuleDict()
        in_channels = 1
        for i in range(self.num_layers):
            out_channels = 2**(6-i//int(np.sqrt(self.num_layers)))
            kernel_size = (vars.N//300)*2**(4 - (i-1)//int(np.sqrt(self.num_layers))) + 1
            # first conv layer
            conv = conv_mish(in_channels, out_channels, kernel_size)
            self.body['layer{}_conv1'.format(i)] = conv
            # second conv layer
            conv = conv_mish(out_channels, out_channels, kernel_size)
            self.body['layer{}_conv2'.format(i)] = conv
            #concatenate
            in_channels = in_channels + out_channels

        #head of size-1 convolutions
        self.head = nn.ModuleDict()
        for i in range(self.num_head_layers):
            out_channels = 2**(3+self.num_head_layers//2 \
                               - i//int(np.sqrt(self.num_head_layers)))
            conv = conv_mish(in_channels, out_channels)
            self.head['layer{}_conv'.format(i)] = conv
            in_channels = out_channels


        #output size-1 convolution
        out_channels = vars.Q
        self.out_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                                  padding=0)
        nn.init.kaiming_uniform_(self.out_conv.weight, nonlinearity='sigmoid')
        nn.init.zeros_(conv.bias)

        #output linear transform params
        output_linear = linspace(-1, 1, vars.N)
        self.register_buffer("output_linear", output_linear)
        

    def forward(self, x):
        #add channel dimension
        x = x.unsqueeze(1)

        #conv layers
        for i in range(self.num_layers):
            dense = x # dense connection
            x = F.mish(self.body['layer{}_conv1'.format(i)](x))
            x = F.mish(self.body['layer{}_conv2'.format(i)](x))
            x = cat((dense, x), dim=1)

        #head conv layers
        for i in range(self.num_head_layers):
            x = F.mish(self.head['layer{}_conv'.format(i)](x))
 
        latent = F.softmax(self.out_conv(x), -1)  

        #output linear transform
        y = torch.matmul(latent, self.output_linear)   

        return y, latent


class residual_encoder(nn.Module):
    def __init__(self, num_layers, num_head_layers):
        super().__init__()

        self.num_layers = num_layers
        self.num_head_layers = num_head_layers

        #Conv layers with concat connections
        self.conv = nn.ModuleDict()
        in_channels = 1
        for i in range(self.num_layers):
            out_channels = 2**(6-i//int(np.sqrt(self.num_layers)))
            # first conv layer
            kernel_size = 2**(4 - (i-1)//2**(4+self.num_layers//2)) + 1
            conv = conv_mish(in_channels, out_channels, kernel_size)
            self.conv['layer{}_conv1'.format(i)] = conv
            # second conv layer
            kernel_size = (kernel_size - 1)//2 + 1
            conv = conv_mish(out_channels, out_channels, kernel_size)
            self.conv['layer{}_conv2'.format(i)] = conv
            # residual resampling
            conv = resample(in_channels, out_channels)
            self.conv['layer{}_resample'.format(i)] = conv
            in_channels = out_channels

        #head of size-1 convolutions
        self.head = nn.ModuleDict()
        for i in range(self.num_head_layers):
            out_channels = 2**(3+self.num_head_layers//2 \
                               - i//int(np.sqrt(self.num_head_layers)))
            conv = conv_mish(in_channels, out_channels)
            self.head.update({'layer{}_conv'.format(i):conv})
            in_channels = out_channels

        #output size-1 convolution
        out_channels = vars.Q
        self.out_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                                  padding='same')
        nn.init.kaiming_uniform_(self.out_conv.weight, nonlinearity='sigmoid')
        nn.init.zeros_(conv.bias)
        
        #output linear transform params
        weights = linspace(-1, 1, vars.N)
        self.output_linear = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        #add channel dimension
        x = x.unsqueeze(1)

        #conv layers
        for i in range(self.num_layers):
            residual = x
            x = F.mish(self.conv['layer{}_conv1'.format(i)](x))
            x = F.mish(self.conv['layer{}_conv2'.format(i)](x))
            x += self.conv['layer{}_resample'.format(i)](residual)
            

        #head conv layers
        for i in range(self.num_head_layers):
            x = F.mish(self.head['layer{}_conv'.format(i)](x))
 
        latent = F.softmax(self.out_conv(x), -1)    

        #output linear transform
        y = torch.matmul(latent, self.output_linear)

        return y, latent


class dilated_encoder(nn.Module):
    def __init__(self, num_layers, num_head_layers, receptive_field=1.0, init_channels_exp=3, channels_growdth = None):
        super().__init__()

        self.num_layers = num_layers
        self.num_head_layers = num_head_layers

        if channels_growdth == None:
            channels_growdth = 1/int(np.sqrt(self.num_layers))
        kernel_vect = get_kernel_sizes(num_layers, receptive_field)

        #Conv layers with concat connections
        self.conv = nn.ModuleDict()
        in_channels = 1
        dilation = 1
        for i in range(self.num_layers):
            kernel = kernel_vect[i]
            out_channels = 2**(init_channels_exp+int(i*channels_growdth))
            # first conv layer
            dilation = dilation*(kernel-1)
            conv = conv_mish(in_channels, out_channels, kernel, dilation)
            self.conv['layer{}_conv'.format(i)] = conv
            in_channels = out_channels

        #head of size-1 convolutions
        self.head = nn.ModuleDict()
        for i in range(self.num_head_layers):
            out_channels = 2**(3+self.num_head_layers//2 \
                               - i//int(np.sqrt(self.num_head_layers)))
            conv = conv_mish(in_channels, out_channels)
            self.head.update({'layer{}_conv'.format(i):conv})
            in_channels = out_channels

        #output size-1 convolution
        out_channels = vars.Q
        self.out_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                                  padding='same')
        nn.init.kaiming_uniform_(self.out_conv.weight, nonlinearity='sigmoid')
        nn.init.zeros_(conv.bias)
        
        #output linear transform params
        weights = linspace(-1, 1, vars.N)
        self.output_linear = nn.Parameter(weights, requires_grad=False)

    def forward(self, x):
        #add channel dimension
        x = x.unsqueeze(1)

        #conv layers
        for i in range(self.num_layers):
            x = F.mish(self.conv['layer{}_conv'.format(i)](x))           

        #head conv layers
        for i in range(self.num_head_layers):
            x = F.mish(self.head['layer{}_conv'.format(i)](x))
 
        latent = F.softmax(self.out_conv(x), -1)     

        #output linear transform
        y = torch.matmul(latent, self.output_linear)

        return y, latent


class decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Atenuations
        self.A = nn.Parameter(torch.ones(vars.Q))
        parametrize.register_parametrization(self, "A", UnitCap())

        # pre narrow conv
        in_channels = vars.Q
        out_channels = vars.Q
        # How to deal with changes in Δλ?
        # kernel_size = int(vars.N*np.max(vars.Δλ)/vars.Δ/4)
        kernel_size = 51
        # self.conv = SymmetricConv1d(in_channels, out_channels, kernel, groups=out_channels)
        self.conv = NarrowConv1D(in_channels, out_channels, kernel_size, groups=out_channels)


        # single spectra
        in_channels = vars.Q
        out_channels = vars.Q
        kernel_size = vars.N + 1
        self.transpose_conv = SymmetricNormConvTranspose1d(in_channels, out_channels,
                                                 kernel_size, bias=False,
                                                 padding=vars.N//2,
                                                 groups=out_channels)
        nn.init.kaiming_uniform_(self.transpose_conv.weight, nonlinearity='linear')

        # joint spectra
        if vars.topology.startswith('serial'):
            # TODO: add warning if not serial_rec or serial??
            # peak reflectance assigned to the last FBG (i=-1)
            def func(x):
                #change batch dim for channel dim
                x = torch.transpose(x, 0, 1)

                r = torch.zeros_like(x[0]) # reflectance
                # t = torch.ones_like(x[0]) # transmittance
                t_2 = torch.ones_like(x[0]) # transmittance squared
                # traverse channel dims
                for r_next, a in zip(x, self.A):
                    t_next = 1 - r_next
                    s = 1/(1 - a*r*r_next) # resonance
                    # r = r + a*r_next*t**2*s
                    # t = torch.sqrt(a)*t*t_next*s
                    r = r + a*r_next*t_2*s
                    t_2 = a*t_2*t_next**2*s**2
                #normalize
                # r = r/(x[-1].max().detach()*self.A.prod())
                # peak = self.transpose_conv.weight[-1,0,vars.N//2].detach()
                peak = self.transpose_conv.weight[-1, 0,vars.N//2]
                r = r/(peak*self.A.prod())
                return r
            self.joint = func
        elif vars.topology == 'parallel':
            self.i = torch.argmax(sim.get_max_R(vars.S)*vars.A)
            def func(x):
                #change batch dim for channel dim
                x = torch.transpose(x, 0, 1)
                # tensor dot along channel dim
                r = torch.tensordot(self.A, x, dims=1)
                #normalize
                r = r/(x[self.i].max().detach()*self.A[self.i])
                return r
            self.joint = func
        else:
            raise ValueError("Topology must be one of {'serial','parallel'}")

    def forward(self, x):

        x = self.conv(x)
        x = F.softmax(x, dim=-1)

        #output transposed conv
        with parametrize.cached(): # To compute parametrization once
            x = self.transpose_conv(x)
            x = self.joint(x)

        return x
        

# ---------------------------------------------------------------------------- #
#                               Lightning Models                               #
# ---------------------------------------------------------------------------- #

class base_model(pl.LightningModule):
    def __init__(self, weights=None, batch_size=1000, lr = 3e-1,
                 data=None, optimizer='adam', optimizer_kwargs={},
                 weight_decay=0, scheduler='one_cycle', scheduler_kwargs={},
                 reduce_on_plateau=False, noise=False, shift_augment = False,
                 encoder_kwargs={}, **kwargs):
        super().__init__()
        
        # Hyperparameters
        self.save_hyperparameters(ignore=['weights', 'data', 'optimizer', 
                                          'optimizer_kwargs','scheduler',
                                          'scheduler_kwargs',
                                          'reduce_on_plateau', 'noise'],
                                  logger=False)

        if weights is None:
            weights = np.full(vars.Q, 1/vars.Q)
        weights = weights/np.sum(weights)
        weights = torch.tensor(weights.copy(), dtype=torch.get_default_dtype())
        self.register_buffer("weights", weights)

        if data is None:
            data = sim.gen_data()
            self.data = sim.normalize(*data)
        else:
            self.data = data
        
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs

        self.reduce_on_plateau= reduce_on_plateau

        self.noise = noise

        self.shift_augment = shift_augment
        self.batch_size = self.hparams.batch_size

        self.param_groups = None

        # get one batch from validation as an example 
        for batch in self.val_dataloader():
            x, y = batch
            self.example_input_array = x
            break

    def setup(self, stage):
        super().setup(stage)

        self.batch_size = self.hparams.batch_size//(self.shift_augment+1)
        if self.hparams.batch_size%(self.shift_augment+1):
            raise ValueError("shift_augment+1 should be a factor of batch_size")

    def metric(self, outputs, targets):
        x, y = targets
        y_hat, latent = outputs
        return F.l1_loss(y, y_hat)*vars.Δ/vars.p

    def l2(self, outputs, targets):
        x, y = targets
        y_hat, latent = outputs
        return torch.norm(latent, p=2, dim=-1).mean()

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, 
                                    {'hp/val_MAE': -1, 'hp/l2': -1})

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        if self.noise:
            if isinstance(self.noise, float):
                sigma = self.noise
            else:
                sigma = 1e-2
            noise = sigma*torch.randn_like(x)
            x += noise
        if self.shift_augment:
            train_batch = batch_shift((x, y), self.shift_augment)
            x, y = train_batch
        outputs = self.forward(x)
        targets = train_batch
        loss = self.loss(outputs, targets)
        self.log('train_loss', loss, prog_bar=True)
        metric = self.metric(outputs, targets)
        self.log('train_MAE', metric, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        outputs = self.forward(x)
        targets = val_batch
        loss = self.loss(outputs, targets)
        self.log('val_loss', loss, prog_bar=True)
        metric = self.metric(outputs, targets)
        self.log('val_MAE', metric, prog_bar=True)
        self.log('hp/val_MAE', metric)
        self.log('hp/l2', self.l2(outputs, targets))

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("loss", None)
        return items

    def _prep_dataloader(self, Z, shuffle=False):
        Z = map(lambda x: tensor(x, dtype=torch.get_default_dtype(), 
                                 device=self.device),
                   Z)
        ds = TensorDataset(*Z)
        return DataLoader(ds, self.batch_size, shuffle)

    def train_dataloader(self):
        X, y = self.data[0:2]
        return self._prep_dataloader((X, y), shuffle=True)

    def val_dataloader(self):
        X, y = self.data[2:]
        return self._prep_dataloader((X, y))

    def configure_optimizers(self):
        if self.param_groups is None:
            params = filter(lambda p: p.requires_grad, self.parameters())
        else:
            params = self.param_groups
        if self.optimizer == 'adam':
            optimizer = Adam(params, lr=self.hparams.lr,
                             weight_decay=self.hparams.weight_decay,
                             **self.optimizer_kwargs)
        elif self.optimizer == 'sgd':
            optimizer = SGD(params, lr=self.hparams.lr,
                            weight_decay=self.hparams.weight_decay, 
                            **self.optimizer_kwargs)
        elif self.optimizer == 'adamw':
            optimizer = AdamW(params, lr=self.hparams.lr, 
                              weight_decay=self.hparams.weight_decay,
                              **self.optimizer_kwargs)
        elif issubclass(self.optimizer, torch.optim.Optimizer):
            optimizer = self.optimizer(params, lr=self.hparams.lr, 
                              weight_decay=self.hparams.weight_decay,
                              **self.optimizer_kwargs)
        else:
            raise ValueError("optimizer should be one of {'adam', 'sgd','adamw'} \
                             or a subclass of Optimizer")

        if self.scheduler is None:
            scheduler = None
            #return optimizer
        elif self.scheduler == 'one_cycle':
            # default kwargs
            scheduler_kwargs = dict(div_factor=1e2, final_div_factor=1e3,
                                    cycle_momentum=True,
                                    anneal_strategy='cos', three_phase=False)
            if self.trainer:
                total_steps = self.trainer.max_epochs*vars.M//self.batch_size
                scheduler_kwargs['total_steps'] = total_steps
            scheduler_kwargs.update(self.scheduler_kwargs)
            scheduler = OneCycleLR(optimizer, 
                                   max_lr=[param_group['lr'] \
                                           for param_group in optimizer.param_groups], 
                                   **scheduler_kwargs)
            scheduler = {"scheduler": scheduler, "interval" : "step" }
        elif issubclass(self.scheduler, torch.optim.lr_scheduler._LRScheduler):
            scheduler = self.scheduler(optimizer, **self.scheduler_kwargs)
            scheduler = {"scheduler": scheduler, "interval" : "step" }
        elif isinstance(self.scheduler, dict):
            scheduler = self.scheduler.copy()
            scheduler['scheduler'] = scheduler['scheduler'](optimizer, 
                                                            **self.scheduler_kwargs)
        else:
            raise ValueError("scheduler should be one of {'one_cycle'} or \
                             a subclass of _LRScheduler")

        if scheduler is None:
            scheduler_list = []
        else:
            scheduler_list = [scheduler]

        if self.reduce_on_plateau:
            if self.reduce_on_plateau is True:
                patience = 1000
            else:
                patience = self.reduce_on_plateau
            scheduler = ReduceLROnPlateau(optimizer, patience=patience, mode='min',
                                          factor=0.9)
            scheduler = dict(scheduler=scheduler, monitor='val_MAE',
                             #reduce_on_plateau=True,
                             strict =  False,
                             interval = "epoch")
            scheduler_list.append(scheduler)
        
        if scheduler_list:
            return [optimizer], scheduler_list
        else:
            return optimizer

    def predict(self, X):
        X = self._prep_dataloader((X,))
        return np.concatenate([self(x[0])[0].detach().cpu().numpy() for x in X])


class encoder_model(base_model):
    def __init__(self, reg=1e-5,
                 rho=1e-1, num_layers=3, num_head_layers=2,
                 encoder_type = 'dense', encoder_kwargs={}, reg_type='l1',
                 prefilter=False, **kwargs):
        super().__init__(reg=reg, rho=rho, num_layers=num_layers,
                         num_head_layers=num_head_layers, 
                         encoder_type=encoder_type,
                         encoder_kwargs=encoder_kwargs,reg_type=reg_type,
                         **kwargs)

        if encoder_type == 'dense':
            encoder = dense_encoder
        elif encoder_type == 'residual':
            encoder = residual_encoder
        elif encoder_type == 'dilated':
            encoder = dilated_encoder
        elif issubclass(encoder_type, nn.Module):
            encoder = encoder_type
        else:
            raise ValueError("encoder_type must be {'dense','residual', 'dilated'} or a subclass of nn.Module")
        self.encoder = encoder(num_layers, num_head_layers, **encoder_kwargs)

        self.filter = prefilter

    def forward(self, x):
        if self.filter:
            x = lowpass_biquad(x, sample_rate=1, cutoff_freq=0.2)
        return self.encoder(x)

    def setup(self, stage):
        super().setup(stage)
        
        if self.hparams.reg_type == None or self.hparams.reg == 0:
            self.reg_func = null
        else:
            if self.hparams.reg_type == 'l1':
                self.reg_func = l1_norm
            elif self.hparams.reg_type == 'kl_div':
                self.reg_func = kl_div(self.hparams.rho)
            elif self.hparams.reg_type == 'spread':
                self.reg_func = spread(self.hparams.sigma)
            elif self.hparams.reg_type == 'kurtosis':
                self.reg_func = kurtosis(self.hparams.kappa)
            elif self.hparams.reg_type == 'gaussianess':
                self.reg_func = gaussianess(self.hparams.nu)
            else:
                raise ValueError('reg_type has to be {l1, kl_div, spread, kurtosis}')

        self.reg_loss = weighted_mse(self.weights)

    def loss(self, outputs, targets):
        x, y = targets
        y_hat, latent = outputs

        loss = self.reg_loss(y, y_hat)\
              + self.hparams.reg*self.reg_func(latent)
        return loss
    
    def batch_forward(self, X):
        shape = X.shape
        X = self._prep_dataloader((X,))
        Y_hat = np.empty((shape[0], vars.Q))
        L = np.empty((shape[0], vars.Q, shape[1]))
        for i, x in enumerate(X):
            y_hat, latent = self(x[0])
            Y_hat[i*self.batch_size: (i+1)*self.batch_size] = y_hat.detach().cpu().numpy()
            L[i*self.batch_size: (i+1)*self.batch_size] = latent.detach().cpu().numpy()
        return Y_hat, L


class autoencoder_model(encoder_model):
    def __init__(self, gamma=1e-3, smooth_reg=1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters('gamma', 'smooth_reg', logger=False)

        self.decoder = decoder()

    def forward(self, x):
        y, latent = self.encoder(x)
        x = self.decoder(latent)
        return x, y, latent

    def setup(self, stage=0):
        super().setup(stage)

        if self.hparams.smooth_reg == 0:
            self.roughness = null
        else:
            self.roughness = roughness

        self.reg_loss = weighted_mse(self.weights)
        self.rec_loss = F.mse_loss

        if self.hparams.gamma == 0:
            self.rec_loss = null
        elif self.hparams.gamma == 1:
            self.reg_loss = null

    def metric(self, outputs, targets):
        """Mean Absolute Value in picometers"""
        x, y = targets
        x_hat, y_hat, latent = outputs
        return F.l1_loss(y, y_hat)*vars.Δ/vars.p

    def l2(self, outputs, targets):
        """l2-norm of latent variable"""
        x, y = targets
        x_hat, y_hat, latent  = outputs
        return torch.norm(latent, p=2, dim=-1).mean()

    def loss(self, outputs, targets):
        x, y = targets
        x_hat, y_hat, latent = outputs
        loss = (1-self.hparams.gamma)*self.reg_loss(y, y_hat) \
               + self.hparams.gamma*self.rec_loss(x_hat, x) \
               + self.hparams.reg*self.reg_func(latent) \
               + self.hparams.smooth_reg \
                 *self.roughness(self.decoder.transpose_conv.weight)
        return loss

    def val_rec_loss(self, outputs, targets):
        """Reconstruction Error"""
        x, y = targets
        x_hat, y_hat, latent = outputs
        return F.mse_loss(x_hat, x)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        outputs = self.forward(x)
        targets = val_batch
        loss = self.loss(outputs, targets)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_rec_loss', self.val_rec_loss(outputs, targets))
        metric = self.metric(outputs, targets)
        self.log('val_MAE', metric, prog_bar=True)
        self.log('hp/val_MAE', metric)
        self.log('hp/l2', self.l2(outputs, targets))

    def predict(self, X):
        X = self._prep_dataloader((X,))
        return np.concatenate([self(x[0])[1].detach().cpu().numpy() for x in X])
    
    def batch_forward(self, X):
        shape = X.shape
        X = self._prep_dataloader((X,))
        X_hat = np.empty(shape)
        Y_hat = np.empty((shape[0], vars.Q))
        L = np.empty((shape[0], vars.Q, shape[1]))
        for i, x in enumerate(X):
            x_hat, y_hat, latent = self(x[0])
            X_hat[i*self.batch_size: (i+1)*self.batch_size] = x_hat.detach().cpu().numpy()
            Y_hat[i*self.batch_size: (i+1)*self.batch_size] = y_hat.detach().cpu().numpy()
            L[i*self.batch_size: (i+1)*self.batch_size] = latent.detach().cpu().numpy()
        return X_hat, Y_hat, L
