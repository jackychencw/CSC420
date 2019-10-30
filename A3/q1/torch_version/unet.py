import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels = 64, n_classes = 2):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.downlayer1 = self.conv(1, self.n_channels)
        self.downlayer2 = self.dconv(self.n_channels, 2 * self.n_channels)
        self.downlayer3 = self.dconv(2 * self.n_channels, 4 * self.n_channels)
        self.downlayer4 = self.dconv(4 * self.n_channels, 8 * self.n_channels)
        self.botlayer = self.botconv(8 * self.n_channels, 16 * self.n_channels)
        self.uplayer1 = self.uconv(16 * n_channels, 4 * self.n_channels)
        self.uplayer2 = self.uconv(8 * n_channels, 2 * self.n_channels)
        self.uplayer3 = self.uconv(4 * n_channels, 1 * self.n_channels)
        self.uplayer4 = self.conv(2 * n_channels, self.n_channels)
        self.final = nn.Conv2d(self.n_channels, self.n_classes, kernel_size = 1)
        self.softmax = nn.LogSoftmax(dim=1)
        
    
    def forward(self, x):
        d1 = self.downlayer1(x)
        d2 = self.downlayer2(d1)
        d3 = self.downlayer3(d2)
        d4 = self.downlayer4(d3)
        b = self.botlayer(d4)
        u1 = torch.cat([d4, b], dim=1)
        u1 = self.uplayer1(u1)
        u2 = torch.cat([d3, u1], dim=1)
        u2 = self.uplayer2(u2)
        u3= torch.cat([d2, u2], dim=1)
        u3 = self.uplayer3(u3)
        u4= torch.cat([d1, u3], dim=1)
        u4 = self.uplayer4(u4)
        output = self.final(u4)
        if self.n_classes > 1:
            output = self.softmax(output)
            return output.squeeze()
        else:
            return torch.sigmoid(output)
    
    def conv(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def dconv(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        return nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def botconv(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        return nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels,in_channels,kernel_size=2,stride=2)
        )
    
    def uconv(self, in_channels, out_channels, kernel_size = 3, padding = 1):
        return nn.Sequential(
            nn.Conv2d(in_channels, 2 * out_channels, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm2d(2 * out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm2d(2 * out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2 * out_channels, out_channels,kernel_size=2,stride=2)
        )

        