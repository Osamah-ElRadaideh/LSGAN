import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch import Tensor
from torch import autograd

class FF(nn.Module):
    def __init__(self, in_dim, out_dim,):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)
    def forward(self, x):
        return self.relu(self.fc(x))
class up_block(nn.Module):
    #up_sampling block of the generator
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,act='relu'):
        super().__init__()
        assert act.lower() in ['relu', 'sigmoid'], f'act must be either relu or sigmoid, got {act} instead'
        self.act = act
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        if self.act =='relu':
            return(self.relu(x))
        else:
            # return x 
            return(self.sigmoid(x))

class down_block(nn.Module):
    #downsampling block of the discriminator
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2,act='relu'):
        super().__init__()
        if kernel_size == 1:
            padding = 0
        else: 
            padding = 1
        assert act.lower() in ['relu', 'none'], f'act must be either relu or none, got {act} instead'
        self.act = act
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True, negative_slope=0.2)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act =='relu':
            return(self.relu(x))
        else:
            return x 
        



class Generator(nn.Module):
    '''
    LSGAN generator
    input: noise of shape (bs,latent_dim)
    output: image of shape (bs, out_channels, 64, 64)
    by default the noise is sampled at (bs, 32, 4,4) and outputs an RGB image (3, 64, 64)
    '''
    def __init__(self, latent_size, channel_size = 256, up_sizes= [4,2,2,4]):
        super().__init__()
        self.channel_size = channel_size
        self.ff = FF(latent_size, 4096)
        self.conv1 = up_block(channel_size, channel_size // 2, kernel_size = 4, stride = up_sizes[0])
        self.conv2 = up_block(channel_size // 2, channel_size // 4, kernel_size = 2, stride=up_sizes[1])
        self.conv3 = up_block(channel_size // 4, channel_size // 8, kernel_size = 2, stride=up_sizes[2])
        self.conv4 = up_block(channel_size // 8 ,3,act='sigmoid')
                              
            
    def forward(self, x):
        x = self.ff(x)
        x = x.view(x.shape[0], self.channel_size, 4, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x 




class Discriminator(nn.Module):
    '''
    Discriminator block
    input: image of shape (bs, in_channels, 64 ,64)
    output: probabilities of shape (bs, 1)
    
    '''
    def __init__(self, channel_size=64, in_channels=3):
        super().__init__()
        self.conv1 = down_block(in_channels,channel_size)
        self.conv2 = down_block(channel_size,2  * channel_size)
        self.conv3 =  down_block(2 * channel_size,4 * channel_size)
        self.conv4 = down_block(4 * channel_size, 8 * channel_size)

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4608, 1)    
    def forward(self, x):
        features = []
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        return self.fc(x), features


'''
LSGAN losses from https://arxiv.org/pdf/1611.04076.pdf
'''



def gen_loss(fake_outs):
    loss = 0.5 * torch.mean((fake_outs - 1) ** 2)

    return loss


def disc_loss(real_outs, fake_outs):
    d_loss = 0.5 * torch.mean((real_outs - 1)**2)
    g_loss = 0.5 * torch.mean(fake_outs ** 2)
    # d_loss = torch.mean((1-real_outs)**2)
    # g_loss = torch.mean(fake_outs**2)
    return d_loss + g_loss

def logistic_g_loss(d_generated_outputs: Tensor) -> Tensor:
    """
    Logistic generator loss.
    Assumes input is D(G(x)), or in our case, D(W(z)).
    `disc_outputs` of shape (bs,)
    """
    # d_generated_outputs = torch.sigmoid(d_generated_outputs)
    # loss = torch.log(1 - d_generated_outputs).mean()
    loss = F.softplus(-d_generated_outputs).mean()
    return loss

def logistic_d_loss(d_real_outputs, d_generated_outputs):
    """
    Logistic discriminator loss.
    `d_real_outputs` (bs,): D(x), or in our case D(c)
    `d_generated_outputs` (bs,): D(G(x)), or in our case D(W(z))
    D attempts to push real samples as big as possible (as close to 1.0 as possible), 
    and push fake ones to 0.0
    """
    # d_real_outputs = torch.sigmoid(d_real_outputs)
    # d_generated_outputs = torch.sigmoid(d_generated_outputs)
    # loss = -( torch.log(d_real_outputs) + torch.log(1-d_generated_outputs) )
    # loss = loss.mean()
    term1 = F.softplus(d_generated_outputs) 
    term2 = F.softplus(-d_real_outputs)
    return (term1 + term2).mean()


def compute_gp(netD, real_data, fake_data):
        batch_size = real_data.size(0)
        # Sample Epsilon from uniform distribution
        eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
        eps = eps.expand_as(real_data)
        
        # Interpolation between real data and fake data.
        interpolation = eps * real_data + (1 - eps) * fake_data
        
        # get logits for interpolated images
        interp_logits = netD(interpolation)
        grad_outputs = torch.ones_like(interp_logits)
        
        # Compute Gradients
        gradients = autograd.grad(
            outputs=interp_logits,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Compute and return Gradient Norm
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2) * 10
