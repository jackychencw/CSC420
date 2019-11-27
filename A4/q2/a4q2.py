from utils import *
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import progressbar
from scipy.ndimage.filters import gaussian_filter

patch_size = 14

f = 721.537700
px = 609.559300
py = 172.854000
baseline = 0.5327119288
x1 = 685
x2 = 804
y1 = 181
y2 = 258


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


left_color = load_color_image('./A4_files/000020_left.jpg')
right_color = load_color_image('./A4_files/000020_right.jpg')
left_bright = increase_brightness(left_color)
right_bright = increase_brightness(right_color)
save_image("./A4_files/000020_left_bright.jpg", left_bright)
save_image("./A4_files/000020_right_bright.jpg", right_bright)
left_grey = load_grey_scale_image('./A4_files/000020_left_bright.jpg')
right_grey = load_grey_scale_image('./A4_files/000020_right_bright.jpg')
# left_grey = gaussian_filter(left_grey, sigma=7)
# right_grey = gaussian_filter(right_grey, sigma=7)


def ssd(patch1, patch2):
    diff = patch1 - patch2
    ssd = np.sum(diff**2)
    return ssd


def nc(patch1, patch2):
    a = np.sum(patch1 * patch2)
    b = np.sum(patch1 ** 2) * np.sum(patch2 ** 2)
    c = a * 1./b
    return c


def draw_box(x1, y1, x2, y2, img, thickness=0):
    if thickness == 0:
        img[y1:y2 + 1, x1] = [0, 255, 0]
        img[y1:y2 + 1, x2] = [0, 255, 0]
        img[y2, x1: x2 + 1] = [0, 255, 0]
        img[y1, x1: x2 + 1] = [0, 255, 0]
    else:
        img[y1:y2 + 1, x1-thickness:x1+thickness + 1] = [0, 255, 0]
        img[y1:y2 + 1, x2-thickness:x2+thickness + 1] = [0, 255, 0]
        img[y2 - thickness:y2 + thickness + 1, x1: x2 + 1] = [0, 255, 0]
        img[y1 - thickness:y1 + thickness + 1, x1: x2 + 1] = [0, 255, 0]
    return img

# (a)


def down_sample(img, factor=2, ite=3):
    for i in range(ite):
        height, width = img.shape[0], img.shape[1]
        img = cv.pyrDown(img, dstsize=(width // factor, height // factor))
    return img


def up_sample(img, factor=2, ite=3):
    for i in range(ite):
        height, width = img.shape[0], img.shape[1]
        img = cv.pyrUp(img, dstsize=(width * factor, height * factor))
    return img


def scan_all(scan_size=50, downsample=False, img1=left_grey, img2=right_grey, patch_size=patch_size):
    if downsample:
        img1 = down_sample(img1)
        img2 = down_sample(img2)
    height1, width1 = img1.shape[0], img1.shape[1]
    height2, width2 = img2.shape[0], img2.shape[1]
    assert height1 == height2 and width1 == width2

    # Setup progress bar
    ite_num = height1 * width1 * 2 * min(scan_size, width2)
    bar = progressbar.ProgressBar(maxval=ite_num,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    count = 0
    # Start sliding window
    diff = np.zeros((height1, width1))
    scores = np.zeros((height1, width1))
    match_map = np.zeros((height1, width1))
    for y1 in range(height1):
        for x1 in range(width1):
            x1i = max(0, x1 - patch_size)
            toleft = abs(x1 - x1i)
            x1d = min(width1, x1 + patch_size)
            toright = abs(x1 - x1d)
            y1i = max(0, y1 - patch_size)
            toup = abs(y1 - y1i)
            y1d = max(height1, y1 + patch_size)
            todown = abs(y1 - y1d)

            patch1 = img1[y1i:y1d, x1i:x1d]
            y2 = y1

            best_score = None
            best_ind = None
            for x2 in range(max(x1 - scan_size, toleft), min(x1 + scan_size + 1, width2 - toright)):
                x2i = x2 - toleft
                x2d = x2 + toright
                y2i = y2 - toup
                y2d = y2 + todown
                patch2 = img2[y2i:y2d, x2i:x2d]
                score = ssd(patch1, patch2)
                if best_score is None or score <= best_score:
                    best_score = score
                    best_ind = x2
                count += 1
                bar.update(count)
            match_map[y1, best_ind] = True
            scores[y1, x1] = abs(best_score)
            diff[y1, x1] = abs(best_ind - x1)
    scores = np.interp(scores, (scores.min(), scores.max()), (0, 5.0))
    diff += scores
    if downsample:
        diff = up_sample(diff)
    return diff


def hconcate(img1, img2):
    h1 = img1.shape[0]
    h2 = img2.shape[0]
    if h1 != h2:
        if h1 > h2:
            temp = np.zeros((h1, img2.shape[1]))
            temp[:h2, :] = img2
            img2 = temp
        else:
            temp = np.zeros((h2, img1.shape[1]))
            temp[:h1, :] = img1
            img1 = temp
    assert img1.shape[0] == img2.shape[0]
    img = np.hstack((img1, img2))
    save_image('./concated.jpg', img)
    return img


def vconcate(img1, img2):
    (h1, w1) = img1.shape
    (h2, w2) = img2.shape
    if w1 != w2:
        if w1 > w2:
            temp = np.zeros((h2, w1))
            temp[:, :w2] = img2
            img2 = temp
        else:
            temp = np.zeros((h1, w2))
            temp[:, :w1] = img1
            img1 = temp
    assert img1.shape[1] == img2.shape[1]
    img = np.vstack((img1, img2))
    return img


def calculate_depth(diff, f=f, T=baseline):
    # where_zero = np.where(diff == 0)
    # where_not_zero = np.where(diff != 0)
    height, width = diff.shape[0], diff.shape[1]
    depths = (f * T) / (diff)
    # depths[where_zero] = depths[where_not_zero].max()
    return depths


if __name__ == "__main__":
    # (a)
    diff = scan_all()
    depth = calculate_depth(diff)
    save_image('./depth.jpg', depth)
    depth2 = np.interp(depth, (depth.min(), depth.max()), (0, 255.0))
    save_image('./depth2.jpg', depth)
    # (b)
