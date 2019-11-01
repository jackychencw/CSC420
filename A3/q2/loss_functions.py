import numpy as np
import tensorflow as tf
import torch
# import sys.float_info.epsilon as epsilon

def sorensen_dice_loss(pred, target):
    return tf.reduce_mean(tf.reduce_mean(tf.abs(pred - target)))

def mse(pred, target):
    return tf.reduce_mean(tf.reduce_mean(
        tf.math.squared_difference(pred, target)
    ))

# def bce(pred, target):
#     return tf.reduce_mean(
#         tf.reduce_sum(
#             -(target * tf.math.log(pred + epsilon) + (1 - target) * tf.math.log(1 - pred + epsilon))
#         )
#     )