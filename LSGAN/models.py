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
        features.append(x)
        x = self.conv2(x)
        features.append(x)
        x = self.conv3(x)
        features.append(x)
        x = self.conv4(x)
        features.append(x)
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
    return d_loss + g_loss

