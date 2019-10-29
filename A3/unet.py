import numpy as np 
import torch 
import torch.nn as nn
from utils import *
from dataset import *
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D

def downconv(input, n_filters, kernel_size=3, batchnorm=True):
    c1 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input)
    if batchnorm:
        c1 = BatchNormalization()(c1)
    c1 = Activation("relu")(c1)
    c2 = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(c1)
    if batchnorm:
        c2 = BatchNormalization()(c2)
    c2 = Activation("relu")(c2)
    return c2

def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # contracting path
    c1 = conv2d_block(input_img, n_filters=n_filters*1,
                      kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2,
                      kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4,
                      kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8,
                      kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters*16,
                      kernel_size=3, batchnorm=batchnorm)

    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3),
                         strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8,
                      kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3),
                         strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4,
                      kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3),
                         strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2,
                      kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3),
                         strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1,
                      kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
class unet(nn.Module):
    def __init__(self, input, kernel_size, padding):
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
