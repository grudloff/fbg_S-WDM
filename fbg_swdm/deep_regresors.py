
from torch import nn, cat, linspace, tensor
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import OneCycleLR
import torch
from torch import Tensor
import pytorch_lightning as pl
import numpy as np

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

def kl_div(hparams):
    def func(input):
        return kl_div_func(hparams.rho, input)
    return func

@torch.jit.script
def spread(input: Tensor) -> Tensor:
    #
    # input: B, C, W tensor
    x = torch.arange(float(vars.N), device = input.device)
    mean = torch.sum(x*input, dim=-1, keepdim=True)
    dist_mean = torch.abs(x - mean) # distance to mean
    dist_mean = dist_mean.detach() #don't propagate gradient through here
    spread = torch.sum(dist_mean*input**2, dim=-1)
    return spread.mean()

@torch.jit.script
def l1_norm(input: Tensor) -> Tensor:
    return torch.norm(input, 1)

# -------------------------------- Model Utils ------------------------------- #

def conv_mish(in_channels, out_channels, kernel_size=1):
    # create & initialize conv layer for mish activation
    conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')
    nn.init.kaiming_uniform_(conv.weight, a=0.0003)
    nn.init.zeros_(conv.bias)
    return conv

def resample(in_channels, out_channels):
    conv = nn.Conv1d(in_channels, out_channels, 1, padding='same', bias=False)
    nn.init.kaiming_uniform_(conv.weight, a=0.0003)
    return conv

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
            kernel_size = 2**(4 - (i-1)//int(np.sqrt(self.num_layers))) + 1
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
        weights = linspace(-1, 1, vars.N)
        self.output_linear = nn.Parameter(weights, requires_grad=False)

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

        #Conv layers with concat connections
        self.conv = nn.ModuleDict()
        in_channels = 1
        for i in range(self.hparams.num_layers):
            out_channels = 2**(6-i//int(np.sqrt(num_layers)))
            # first conv layer
            kernel_size = 2**(4 - (i-1)//2**(4+num_layers//2)) + 1
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
        for i in range(self.hparams.num_head_layers):
            out_channels = 2**(3+self.hparams.num_head_layers//2 \
                               - i//int(np.sqrt(self.hparams.num_head_layers)))
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
        for i in range(self.hparams.num_layers):
            residual = x
            x = F.mish(self.conv['layer{}_conv1'.format(i)](x))
            x = F.mish(self.conv['layer{}_conv2'.format(i)](x))
            x += self.conv['layer{}_resample'.format(i)](residual)
            

        #head conv layers
        for i in range(self.hparams.num_head_layers):
            x = F.mish(self.head['layer{}_conv'.format(i)](x))
 
        latent = F.softmax(self.out_conv(x), -1)    

        #output linear transform
        y = torch.matmul(latent, self.output_linear)

        return y, latent


class decoder(nn.Module):
    def __init__(self):
        #output transposed conv
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

        x_prev = torch.zeros_like(x[0])
        # traverse channel dims
        for x_next in x:
            x_prev = x_prev + (1-x_prev)**2*x_next/(1-x_next*x_prev)

        x = x_prev
        return x

# ---------------------------------------------------------------------------- #
#                               Lightning Models                               #
# ---------------------------------------------------------------------------- #

class base_model(pl.LightningModule):
    def __init__(self, weights=None, batch_size=1000, lr = 3e-1,
                 data=None, optimizer='adam', scheduler='one_cycle', 
                 scheduler_kwargs = {}, **kwargs):
        super().__init__()
        
        # Hyperparameters
        self.save_hyperparameters(ignore=['weights', 'data', 'optimizer', 
                                          'scheduler', 'scheduler_kwargs'],
                                  logger=False)

        if weights is None:
            weights = np.full(vars.Q, 1/vars.Q)
            weights = weights/np.sum(weights)
            weights = torch.tensor(weights.copy(), dtype=torch.get_default_dtype())
        self.register_buffer("weights", weights)

        if data is None:
            data = sim.gen_data()
            self.data=sim.normalize(*data)
        else:
            self.data = data
        
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.scheduler_kwargs = scheduler_kwargs

        # get one batch from validation as an example 
        for batch in self.val_dataloader():
            x, y = batch
            self.example_input_array = x
            break

    def metric(self, outputs, targets):
        outputs, _ = outputs
        return F.l1_loss(outputs, targets)

    def l2(self, outputs, targets):
        _, latents = outputs
        return torch.norm(latents, p=2, dim=-1).mean()

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, 
                                    {'hp/val_MAE': -1, 'hp/l2': -1})

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        outputs = self.forward(x)
        loss = self.loss(outputs, y)
        self.log('train_loss', loss, prog_bar=True)
        metric = self.metric(outputs, y)
        self.log('train_MAE', metric, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        outputs = self.forward(x)
        loss = self.loss(outputs, y)
        self.log('val_loss', loss, prog_bar=True)
        metric = self.metric(outputs, y)
        self.log('val_MAE', metric, prog_bar=True)
        self.log('hp/val_MAE', metric)
        self.log('hp/l2', self.l2(outputs, y))

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        items.pop("loss", None)
        return items

    def train_dataloader(self):
        X_train, y_train = self.data[0:2]
        X_train, y_train = map(lambda x: tensor(x,
                                                dtype=torch.get_default_dtype(),
                                                ),
                               (X_train, y_train))
        train_ds = TensorDataset(X_train, y_train)
        return DataLoader(train_ds, self.hparams.batch_size, shuffle=True)

    def val_dataloader(self):
        X_test, y_test =self.data[2:]
        X_test, y_test = map(lambda x: tensor(x,
                                                dtype=torch.get_default_dtype(),
                                                ),
                               (X_test, y_test))
        test_ds = TensorDataset(X_test, y_test)
        return DataLoader(test_ds, self.hparams.batch_size)

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        elif self.optimizer == 'sgd':
            optimizer = SGD(self.parameters(), lr=self.hparams.lr)
        else:
            raise ValueError("optimizer should be one of {'adam', 'sgd'}")
        if self.scheduler is None:
            return optimizer
        elif self.scheduler == 'one_cycle':
            scheduler = OneCycleLR(optimizer, max_lr=self.hparams.lr, 
                                   **self.scheduler_kwargs)
            scheduler = {"scheduler": scheduler, "interval" : "step" }
        else:
            raise ValueError("scheduler should be one of {'one_cycle'}")
        return [optimizer], [scheduler]

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.get_default_dtype(), device=self.device)
        x, _ = self.forward(x)
        return x.detach().cpu().numpy()


class encoder_model(base_model):
    def __init__(self, reg=1e-5,
                 rho=1e-1, num_layers=3, num_head_layers=2,
                 encoder_type = 'dense', reg_type='l1', **kwargs):
        super().__init__(reg=reg, rho=rho, num_layers=num_layers,
                         num_head_layers=num_head_layers, 
                         encoder_type=encoder_type, reg_type=reg_type, **kwargs)

        if encoder_type == 'dense':
            self.encoder = dense_encoder(num_layers, num_head_layers)
        elif encoder_type == 'residual':
            self.encoder = residual_encoder(num_layers, num_head_layers)

        if reg_type == 'l1':
            self.reg_func = l1_norm
        elif reg_type == 'kl_div':
            self.reg_func = kl_div(self.hparams)
        elif reg_type == 'spread':
            self.reg_func = spread
        else:
            raise ValueError('reg_type has to be {l1, kl_div, spread}')

    def forward(self, x):
        return self.encoder(x)

    def loss(self, outputs, targets):
        y = targets
        y_hat, latent = outputs

        # weighted regression loss
        reg_loss = weighted_mse(self.weights, y, y_hat)

        loss = reg_loss + self.hparams.reg*self.reg_func(latent)
        return loss


class autoencoder_model(encoder_model):
    def __init__(self, gamma=1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters('gamma', logger=False)

        self.decoder = decoder()

    def forward(self, x):
        y, latent = self.encoder(x)
        x = self.decoder(latent)
        return x, y, latent

    def loss(self, outputs, targets):
        x, y = targets
        x_hat, y_hat, latent = outputs

        # weighted regression loss
        reg_loss = weighted_mse(self.weights, y, y_hat)

        # reconstruction loss
        rec_loss = F.mse_loss(x_hat, x)


        loss = (1-self.hparams.gamma)*reg_loss \
               + self.hparams.gamma*rec_loss \
               + self.hparams.reg*self.reg_func(latent)
        return loss