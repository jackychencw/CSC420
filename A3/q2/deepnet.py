from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Activation, concatenate, Dropout

# Create model
def neural_net(input_dim = 40000,depth=10, n_hidden_1 = 100, n_hidden_2 = 100,  num_classes = 3):
    return Sequential([
        Dense(200, input_dim=input_dim), 
        Activation('relu'), 
        Dropout(0.2), 
        Dense(num_classes)
    ])