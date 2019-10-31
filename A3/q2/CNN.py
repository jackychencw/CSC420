import numpy as np 
import torch 
import torch.nn as nn
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, Dense, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential



def cnn(n_channels=8, out_class = 3, img_size = 200, pool_size=2, depth=1, kernel_size=3):
    print("CNN")
    return Sequential([
        Conv2D(n_channels, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",padding="same"),
        MaxPooling2D((pool_size, pool_size)),
        Flatten(),
        Dense(out_class, activation='softmax')
    ])
