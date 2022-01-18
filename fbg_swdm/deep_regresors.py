
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

import fbg_swdm.simulation as sim
import fbg_swdm.variables as vars

# ---------------------------- Model Loading Utils --------------------------- #

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
def weighted_mse(weights, y, y_hat):
    # weighted regression loss
    reg_loss = torch.dot(weights,torch.mean(F.mse_loss(y_hat, y, 
                                                            reduction='none'), 
                                            dim=0))
    return reg_loss

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
def kurtosis(input: Tensor) -> Tensor:
    x = torch.arange(input.size(-1), device = input.device)
    mean = torch.sum(x*input, dim=-1, keepdim=True)
    dist_mean = x - mean
    K = torch.sum(dist_mean**4*input, dim=-1)/torch.sum(dist_mean**2*input, dim=-1)**2
    return -K.mean()

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
    while True:
        receptive_field = test_kernel(kernel_vect)
        
        if receptive_field > target:
            break
        else:
            kernel_vect[0] = (kernel_vect[0]+1)*2 -1
    if verbose:
        print('kernel_vect: ', kernel_vect)
        print('receptive_field: ', receptive_field)
    
    # if receptive_field surpasses target without increasing kernel
    if kernel_vect[0]==3:
        if verbose:
            print('Too many layers')
        return kernel_vect

    # Decrease first kernel by roughly 2e-1 times and compensate by
    # incresing by two the following kernels to get a receptive_field
    # closer to the target
    best_kernel_vect = kernel_vect.copy()
    best_receptive_field = receptive_field
    
    change_idx = 1
    stop = n_layers
    while True:
        kernel_vect[0] = (kernel_vect[0]+1)//2 -1
        kernel_vect[change_idx] += 2
        receptive_field = test_kernel(kernel_vect)
        if verbose:
            print('kernel_vect: ', kernel_vect)
            print('receptive_field: ', receptive_field)
        if kernel_vect[0]<7 or kernel_vect[0]<kernel_vect[1]:
            break
        if abs(receptive_field-target)<abs(best_receptive_field-target):
            best_kernel_vect = kernel_vect.copy()
            best_receptive_field = receptive_field
            
        change_idx += 1
        if change_idx == stop:
            change_idx = 1
    
    if verbose:
        print('kernel_vect: ', best_kernel_vect)
        print('receptive_field: ', best_receptive_field)
    return best_kernel_vect

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
        #output transposed conv
        self.A = nn.Parameter(torch.ones(vars.Q))
        in_channels = vars.Q
        out_channels = vars.Q
        kernel_size = vars.N + 1
        self.transpose_conv = nn.ConvTranspose1d(in_channels, out_channels,
                                                 kernel_size, bias=False,
                                                 padding=vars.N//2,
                                                 groups=vars.Q)
        nn.init.kaiming_uniform_(self.transpose_conv.weight, nonlinearity='linear')

    def forward(self, x):
        #output transposed conv
        x = self.transpose_conv(x)
        x = torch.transpose(x, 0, 1) #change batch dim for channel dim

        r = torch.zeros_like(x[0])
        t = torch.ones_like(x[0])
        # traverse channel dims
        at = 1.0
        for r_next, a in zip(x, self.A):
            t_next = 1 - r_next
            at = a/at
            F = 1/(1 - at*r*r_next) # resonance
            r = r + at*r_next*t**2*F
            t = torch.sqrt(at)*t*t_next*F

        x = r
        return x
        

# ---------------------------------------------------------------------------- #
#                               Lightning Models                               #
# ---------------------------------------------------------------------------- #

class base_model(pl.LightningModule):
    def __init__(self, weights=None, batch_size=1000, lr = 3e-1,
                 data=None, optimizer='adam', optimizer_kwargs={},
                 weight_decay=0, scheduler='one_cycle', scheduler_kwargs={},
                 reduce_on_plateau=False, noise=False,
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

        # get one batch from validation as an example 
        for batch in self.val_dataloader():
            x, y = batch
            self.example_input_array = x
            break

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
            sigma = sigma*torch.rand((x.size(0), 1), dtype=x.dtype,
                                         layout=x.layout, device=x.device)
            noise = sigma*torch.randn_like(x)
            x += noise
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
        return DataLoader(ds, self.hparams.batch_size, shuffle)

    def train_dataloader(self):
        X, y = self.data[0:2]
        return self._prep_dataloader((X, y), shuffle=True)

    def val_dataloader(self):
        X, y = self.data[2:]
        return self._prep_dataloader((X, y))

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
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
            scheduler_kwargs = dict(div_factor=1, final_div_factor=1e2,
                        cycle_momentum=True,
                        anneal_strategy='cos', three_phase=True)
            scheduler_kwargs.update(self.scheduler_kwargs)
            scheduler = OneCycleLR(optimizer, max_lr=self.hparams.lr, 
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
        X = self.prep_dataloader((X,))
        return np.concatenate([self(x[0])[0].detach().cpu().numpy() for x in X])


class encoder_model(base_model):
    def __init__(self, reg=1e-5,
                 rho=1e-1, num_layers=3, num_head_layers=2,
                 encoder_type = 'dense', encoder_kwargs={}, reg_type='l1',
                 **kwargs):
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

    def forward(self, x):
        return self.encoder(x)

    def setup(self, stage):
        super().setup(stage)
    
        if self.hparams.reg_type == 'l1':
            self.reg_func = l1_norm
        elif self.hparams.reg_type == 'kl_div':
            self.reg_func = kl_div(self.hparams.rho)
        elif self.hparams.reg_type == 'spread':
            self.reg_func = spread
        elif self.hparams.reg_type == 'kurtosis':
            self.reg_func = kurtosis
        else:
            raise ValueError('reg_type has to be {l1, kl_div, spread, kurtosis}')

    def loss(self, outputs, targets):
        x, y = targets
        y_hat, latent = outputs

        # weighted regression loss
        reg_loss = weighted_mse(self.weights, y, y_hat)

        loss = reg_loss + self.hparams.reg*self.reg_func(latent)
        return loss


class autoencoder_model(encoder_model):
    def __init__(self, gamma=1e-3, smooth_reg=1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters('gamma', 'smooth_reg', logger=False)

        self.decoder = decoder()

    def forward(self, x):
        y, latent = self.encoder(x)
        x = self.decoder(latent)
        return x, y, latent

    def metric(self, outputs, targets):
        x, y = targets
        x_hat, y_hat, latent = outputs
        return F.l1_loss(y, y_hat)*vars.Δ/vars.p

    def l2(self, outputs, targets):
        x, y = targets
        x_hat, y_hat, latent  = outputs
        return torch.norm(latent, p=2, dim=-1).mean()

    def loss(self, outputs, targets):
        x, y = targets
        x_hat, y_hat, latent = outputs

        # weighted regression loss
        reg_loss = weighted_mse(self.weights, y, y_hat)

        # reconstruction loss
        rec_loss = F.mse_loss(x_hat, x)

        loss = (1-self.hparams.gamma)*reg_loss \
               + self.hparams.gamma*rec_loss \
               + self.hparams.reg*self.reg_func(latent) \
               + self.hparams.smooth_reg* \
                 roughness(self.decoder.transpose_conv.weight)
        return loss

    def rec_loss(self, outputs, targets):
        x, y = targets
        x_hat, y_hat, latent = outputs
        return F.mse_loss(x_hat, x)

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        outputs = self.forward(x)
        targets = val_batch
        loss = self.loss(outputs, targets)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_rec_loss', self.rec_loss(outputs, targets))
        metric = self.metric(outputs, targets)
        self.log('val_MAE', metric, prog_bar=True)
        self.log('hp/val_MAE', metric)
        self.log('hp/l2', self.l2(outputs, targets))

    def predict(self, X):
        X = self._prep_dataloader((X,))
        return np.concatenate([self(x[0])[1].detach().cpu().numpy() for x in X])