import torch.nn as nn
import torch

class FF(nn.Module):
    def __init__(self, in_dim, out_dim,):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
    def forward(self, x):
        return self.bn(self.fc(x))

class up_block(nn.Module):
    #up_sampling block of the generator
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,act='relu'):
        super().__init__()
        assert act.lower() in ['relu', 'tanh'], f'act must be either relu or tanh, got {act} instead'
        self.act = act
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act =='relu':
            return(self.relu(x))
        else:
            # return x 
            return(self.tanh(x))

class down_block(nn.Module):
    #downsampling block of the discriminator
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2,act='relu'):
        super().__init__()
        assert act.lower() == 'relu' or act is None, f'act must be either relu or None, got {act} instead'
        self.act = act
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
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
    def __init__(self, in_channels=256, out_channels=3, up_sizes= [4,2,2,4]):
        super().__init__()
        self.fc = FF(1024, 4096)
        self.conv1 = up_block(256, 256)
        self.conv2 = up_block( 256, 256, kernel_size = 4, stride = up_sizes[0])
        self.conv3 = up_block(256, 256)
        self.conv4 = up_block(256, 256, kernel_size = 2, stride=up_sizes[1])
        self.conv5 = up_block(256, 128)
        self.conv6 = up_block(128, 64, kernel_size = 2, stride=up_sizes[2])
        self.conv7 = up_block(64,3,act='tanh')
                              
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return x 
    

class Discriminator(nn.Module):
    '''
    Discriminator block
    input: image of shape (bs, in_channels, 64 ,64)
    output: probabilities of shape (bs, 1)
    
    '''
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = down_block(in_channels,256)
        self.conv2 =  down_block(256,256)
        self.conv3 = down_block(256,64)
        self.conv4 = down_block(64,1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16, 1)    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(self.flatten(x))
        return x


'''
LSGAN losses from https://arxiv.org/pdf/1611.04076.pdf
'''


def gen_loss(fake_outs):
    loss = 0.5 * torch.mean((fake_outs - 1) ** 2)

    return loss


def disc_loss(real_outs, fake_outs):
    d_loss = 0.5 * torch.mean((real_outs-1)**2)
    g_loss = 0.5 * torch.mean(fake_outs ** 2)

    return d_loss + g_loss


