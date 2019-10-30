import numpy as np
import cv2 as cv
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt

TRAIN_PATH = "./data/train/"


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


def data_prep(dataset_size=10000, img_size=200):
    X = np.zeros((dataset_size, img_size, img_size, 1))
    Y = np.zeros((dataset_size, img_size, img_size, 3))
    for _ in range(dataset_size):
        target, img, noised_img = data_prep_noisy_circle(img_size, 50, 2)
        noised_img = np.interp(
            noised_img, (noised_img.min(), noised_img.max()), (0, 255.0))
        img = np.interp(img, (img.min(), img.max()), (0, 255.0))
        cv.imwrite("{}input/input.{}.jpg".format(TRAIN_PATH, _), noised_img)
        cv.imwrite("{}target/target.{}.jpg".format(TRAIN_PATH, _), img)


if __name__ == "__main__":
    # data_prep()
