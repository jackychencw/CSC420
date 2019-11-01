import numpy as np 
import torch 
import torch.nn as nn
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate
from tensorflow.keras.models import Model


def conv(input, out_channels, kernel_size, batchnorm = True):
    c1 = Conv2D(filters=out_channels, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",padding="same")(input)
    if batchnorm:
        c1 = BatchNormalization()(c1)
    c1 = Activation("relu")(c1)
    c2 = Conv2D(filters=out_channels, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",padding="same")(c1)
    if batchnorm:
        c2 = BatchNormalization()(c2)
    c2 = Activation("relu")(c2)
    return c2

def UNet(input, n_channels=64, kernel_size=3, batchnorm = True):
    # Down Conv
    c1 = conv(input, n_channels, kernel_size, batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = conv(p1, 2 * n_channels, kernel_size, batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = conv(p2, 4 * n_channels, kernel_size, batchnorm)
    
    # Up Conv
    kernel = (2,2)
    stride = (2,2)

    u8 = Conv2DTranspose(2 * n_channels, kernel, strides=stride, padding='same')(c3)
    u8 = concatenate([u8, c2])
    c8 = conv(u8, 2 * n_channels, kernel_size, batchnorm)
    u9 = Conv2DTranspose(n_channels, kernel, strides=stride, padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = conv(u9, n_channels, kernel_size, batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input], outputs=[outputs])
    return model