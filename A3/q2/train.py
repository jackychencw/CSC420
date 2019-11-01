
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.models import Model, load_model
import dataset
import unet
from skimage.transform import resize
from skimage.io import imread, imshow, concatenate_images
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K


import os
import main
import cv2 as cv
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

im_width = 200
im_height = 200
border = 5



def train_model(model, path_train, augment=False,n_epochs = 100, 
save_path='./weights/weight.h5', batch_size = 30, learning_rate=0.01, 
momentum=0.9, loss="binary_crossentropy"):
    train_dataset = dataset.CatDataset(path_train, im_height, im_width)
    if augment:
        train_dataset.augment()
    X_train, y_train = train_dataset.X, train_dataset.Y
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss, 
    metrics=["accuracy"])
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(save_path, monitor='loss', mode='min',
                        verbose=1, save_best_only=True, save_weights_only=True)
    ]
    results = model.fit(X_train, y_train, batch_size=batch_size,
                        epochs=n_epochs, callbacks=callbacks,validation_split=0.3)
    
    return results

def test_model(x, y, targets, loss, threshold, path_test):
    threshold = threshold
    X_test, y_test = x, y
    model.load_weights(weight_path)
    pred_test = model.predict(X_test, verbose=1)
    iou_sum = 0
    for i in range(pred_test.shape[0]):
        test_pred = pred_test[i]
        t = targets[i]
        test_pred[test_pred>threshold] = 255
        test_pred[test_pred<=threshold] = 0
        save_img("./run2/pred_x{}.jpg".format(i), X_test[i])
        save_img("./run2/pred_{}.jpg".format(i), test_pred)

        source = X_test[i].copy()
        pred = test_pred.copy()
        circles = cv.HoughCircles(pred, cv.HOUGH_GRADIENT,3, 10)
        if circles is not None:
            (r,c,rad) = circles[0]
            main.draw_circle(source, r, c, rad)
        else:
            (r,c,rad) = (0,0,0)
        (r2, c2, rad2) = t
        a = r, c, rad
        b = r2, c2, rad2
        iou_sum += main.iou(a, b)
        output = np.concatenate((X_test[i], pred, source), axis=1)
        # cv.imwrite("./result/r.{}.jpg".format(i), output)

    print("mean iou is {}".format(1. * iou_sum/i))


    model.compile(optimizer=Adam(learning_rate=learning_rate), 
    loss=loss, 
    metrics=["accuracy"])
    model.evaluate(X_test, y_test)

if __name__ == "__main__":
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-train", "--train", type=bool, default = True,
    #     help="Training set")
    # ap.add_argument("-test", "--test", type=bool, default=False,
    #     help="Testing set")
    # args = vars(ap.parse_args())

    # # grab the number of GPUs and store it in a conveience variable
    # train = args["train"]
    # test = args["test"]
    
    train = False
    test = True
    augment = False
    batch_size = 5
    learning_rate = 0.01
    momentum = 0.9
    n_epochs = 100
    path_train = './circle_data/Train/'
    path_test = './circle_data/Test/'
    path_val = './circle_data/Validation/'
    # X, masks, targets = main.data_prep(path_train)
    X_test, masks_test, targets_test = main.data_prep(path_test, dataset_size = 40)
    # X_val, masks_val, targets_val = main.data_prep(path_val, dataset_size = 300)
    loss = "binary_crossentropy"
    
    if augment:
        save_path = "./aug_weights/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path += 'weight_{}.h5'.format(loss)
        out_folder = "./pred_aug_{}".format(loss)
    else:
        save_path = "./weights/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path += 'weight_{}.h5'.format(loss)
        out_folder = "./pred_{}".format(loss)
    
    
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    input_img = Input((im_height, im_width, 1), name='img')
    model = unet.UNet(input_img)
    
    # Train
    if train:
        train_model(model, path_train, n_epochs = n_epochs, augment=False, batch_size = batch_size, learning_rate = learning_rate,momentum=momentum,save_path=save_path, loss=loss)
    # Test
    if test:
        weight_path = save_path
        threshold = 0.3
        test_model(X_test, masks_test, targets_test, loss, threshold, path_test)
