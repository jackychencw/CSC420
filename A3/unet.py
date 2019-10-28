import numpy as np 
import torch 
import torch.nn as nn
# from keras import backend as K
from utils import *
from dataset import *

class unet(nn.Module):
    def __init__(self, kernel_size, padding):
        super(unet, self).__init__()
        base = 64
        self.kernel_size = kernel_size
        self.downconv1 = nn.Sequential(
            nn.Conv2d(1, base, kernel_size = kernel_size, padding = padding),
            nn.ReLU(),
            nn.Conv2d(base, base, kernel_size = kernel_size, padding = padding),
            nn.ReLU()
        )
        self.downconv2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base, 2 * base, kernel_size = kernel_size, padding = padding),
            nn.ReLU(),
            nn.Conv2d(2 * base, 2 * base, kernel_size = kernel_size, padding = padding),
            nn.ReLU()
        )
        self.downconv3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(2 * base, 4 * base, kernel_size = kernel_size, padding = padding),
            nn.ReLU(),
            nn.Conv2d(4 * base, 4 * base, kernel_size = kernel_size, padding = padding),
            nn.ReLU()
        )
        self.downconv4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(4 * base, 8 * base, kernel_size = kernel_size, padding = padding),
            nn.ReLU(),
            nn.Conv2d(8 * base, 8 * base, kernel_size = kernel_size, padding = padding),
            nn.ReLU()
        )
        self.downconv5 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(8 * base, 16 * base, kernel_size = kernel_size, padding = padding),
            nn.ReLU(),
            nn.Conv2d(16 * base, 16 * base, kernel_size = kernel_size, padding = padding),
            nn.ReLU()
        )
        self.upconv1 = upconv(16 * base, 8 * base)
        self.upconv2 = upconv(8 * base, 4 * base)
        self.upconv3 = upconv(4 * base, 2 * base)
        self.upconv4 = upconv(2 * base, base)
        self.final = nn.Conv2d(base, 2, kernel_size = self.kernel_size)

    def forward(self, x):
        step1 = self.downconv1(x)
        step2 = self.downconv2(step1)
        step3 = self.downconv3(step2)
        step4 = self.downconv4(step3)
        step5 = self.downconv5(step4)
        step6 = self.upconv1(step4, step5)
        step7 = self.upconv2(step3, step6)
        step8 = self.upconv3(step2, step7)
        step9 = self.upconv4(step1, step8)
        result = self.finalconv(step9)
    
class upconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super(upconv, self).__init__()
        self.kernel_size = kernel_size
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = self.kernel_size),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size = self.kernel_size),
            nn.ReLU()
        )

    def forward(self, prev, cur):
        upsampled = self.up(cur)
        combined = torch.cat([prev, upsampled], dim=1)
        combined = self.conv(combined)
        return new
