import os, shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, save_img, array_to_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from skimage.transform import resize


TRAIN_PATH = "./circle_data/Train/"
TEST_PATH = "./circle_data/Test/"
IMG_SIZE = 200

def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
        (rr >= 0) &
        (rr < img.shape[0]) &
        (cc >= 0) &
        (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def data_prep_noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    noised_img = img + noise * np.random.rand(*img.shape)
    return (row, col, rad), img, noised_img


def find_circle(img):
    # Fill in this function
    return 100, 100, 30


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
        shape0.intersection(shape1).area /
        shape0.union(shape1).area
    )


def main():
    results = []
    for _ in range(1000):
        params, img = noisy_circle(200, 50, 2)
        detected = find_circle(img)
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).mean())

def clean_dir(path):
    if os.path.exists(path):
        for the_file in os.listdir(path):
            file_path = os.path.join(path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def data_prep(path, dataset_size=1000, img_size=IMG_SIZE):
    # make_dir(path)
    input_path = path + "input/"
    mask_path = path + "mask/"
    # make_dir(input_path)
    clean_dir(input_path)
    # make_dir(mask_path)
    clean_dir(mask_path)
    # print("All files cleaned up")
    X = np.zeros((dataset_size, img_size, img_size, 1))
    masks = np.zeros((dataset_size, img_size, img_size, 1))
    targets = np.zeros((dataset_size, 3))

    for _ in range(dataset_size):
        target, img, noised_img = data_prep_noisy_circle(img_size, 50, 2)
        # noised_img = np.interp(
        #     noised_img, (noised_img.min(), noised_img.max()), (0, 255))
        noised_img = resize(noised_img, (img_size, img_size, 1))
        # img = np.interp(img, (img.min(), img.max()), (0, 255))
        img = resize(img, (img_size, img_size, 1))
        X[_] = noised_img
        masks[_] = img
        targets[_] = target
        save_img("{}input.{}.jpg".format(input_path, _), noised_img)
        save_img("{}target.{}.jpg".format(mask_path, _), img)
    return X, masks, targets


def load_data(path, img_size = IMG_SIZE):
    input_path = path + "input/"
    target_path = path + "target/"
    dataset_size = len(os.listdir(input_path))
    X = np.zeros((dataset_size, img_size, img_size, 1))
    Y = np.zeros((dataset_size, img_size, img_size, 1))
    for _ in range(dataset_size):
        i = load_img(input_path + os.listdir(input_path)[_])
        t = load_img(target_path + os.listdir(target_path)[_])
        i = img_to_array(i)
        t = img_to_array(t)
        i = resize(i, (img_size, img_size, 1))
        t = resize(t, (img_size, img_size, 1))
        X[_] = i
        Y[_] = t
    print("Done loading data")
    return X,Y

def train_model(model, X, Y, test_images, test_labels, save_path, learning_rate = 0.01, momentum = 0.9, loss="categorical_crossentropy"):
    print("Start training model")
    model.compile(optimizer=Adam(learning_rate=learning_rate,
                                momentum=momentum), loss=loss, metrics=["accuracy"])
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(save_path, monitor='accuracy', mode='max',
                        verbose=1, save_best_only=True, save_weights_only=True)
    ]
    results = model.fit(X, to_categorical(Y), batch_size=10,
                        epochs=100, callbacks=callbacks,validation_data=(test_images, to_categorical(test_labels)))
    return results

if __name__ == "__main__":
    # X_train, masks_train, targets_train = data_prep(TRAIN_PATH, dataset_size = 300)
    # X_test, masks_test, targets_test = data_prep(TEST_PATH, dataset_size = 40)
    None