from utils import *
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import progressbar
from scipy.ndimage.filters import gaussian_filter
import math

patch_size = 4

f = 721.537700
px = 609.559300
py = 172.854000
T = 0.5327119288
x1l = 685
x2l = 804
y1l = 181
y2l = 258


def load_color_image(filepath):
    image = cv.imread(filepath)
    return image


def load_grey_scale_image(filepath):
    image = cv.imread(filepath, 0)
    return image


def increase_brightness(img, value=50):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img


def show_image(img):
    cv.imshow("Showing image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def save_image(fname, img):
    cv.imwrite(fname, img)


def draw_box(x1, y1, x2, y2, img, thickness=0):
    img[y1:y2 + 1, x1-thickness:x1+thickness + 1] = [0, 255, 0]
    img[y1:y2 + 1, x2-thickness:x2+thickness + 1] = [0, 255, 0]
    img[y2 - thickness:y2 + thickness + 1, x1: x2 + 1] = [0, 255, 0]
    img[y1 - thickness:y1 + thickness + 1, x1: x2 + 1] = [0, 255, 0]
    return img


if __name__ == '__main__':
    depth = load_grey_scale_image('./000020.png')

    left_color = load_color_image('./A4_files/000020_left.jpg')
    depth = np.interp(depth, (depth.min(), depth.max()), (0.0, 10))

    x1r = math.floor(f * T / depth[y1l, x1l] + px)
    x2r = math.floor(f * T / depth[y2l, x1l] + px)
    y1r = y1l
    y2r = y2l
    new_img = draw_box(x1r, y1r, x2r, y2r, left_color)
    save_image('./out.jpg', new_img)
