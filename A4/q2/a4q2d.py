from utils import *
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import progressbar
from scipy.ndimage.filters import gaussian_filter
import math
from a4q2 import *

patch_size = 200

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


left_color = load_color_image('./A4_files/000020_left_bright.jpg')
right_color = load_color_image('./A4_files/000020_right_bright.jpg')

left_grey = load_grey_scale_image('./A4_files/000020_left_bright.jpg')
right_grey = load_grey_scale_image('./A4_files/000020_right_bright.jpg')


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


def ssd(patch1, patch2):
    diff = patch1 - patch2
    ssd = np.sum(diff**2)
    return ssd


def nc(patch1, patch2):
    a = np.sum(patch1 * patch2)
    b = np.sum(patch1 ** 2) * np.sum(patch2 ** 2)
    c = a * 1./b
    return c


def calculate_3d(depth, img, x1l, y1l, x2l, y2l, x1r, y1r, x2r, y2r):
    Z = np.copy(depth[y1l:y2l, x1l:x2l])
    # Z_center = Z[math.floor(Z.shape[0]/2), math.floor(Z.shape[1]/2)]
    x = np.ones((Z.shape[0], 1)) * np.arange(x1l, x2l)
    x1 = np.arange(x1l, x2l)
    print(x == x1)
    y = np.transpose(np.ones((Z.shape[1], 1))
                     * np.arange(y1l, y2l))
    X = Z * (x - px) * 1. / f
    Y = Z * (y - py) * 1. / f
    X_center = X[math.floor(X.shape[0]/2), math.floor(X.shape[1]/2)]
    Y_center = Y[math.floor(Y.shape[0]/2), math.floor(Y.shape[1]/2)]

    norm = np.sqrt((X - X_center) ** 2 + (Y - Y_center)
                   ** 2 + (Z - Z_center) ** 2)
    minmaxX = np.array([np.min(X[norm < threshold]),
                        np.max(X[norm < threshold])])
    minmaxY = np.array([np.min(Y[norm < threshold]),
                        np.max(Y[norm < threshold])])
    minmaxZ = np.array([np.min(Z[norm < threshold]),
                        np.max(Z[norm < threshold])])
    coords = np.array(np.meshgrid(minmaxX, minmaxY, minmaxZ)).T.reshape(-1, 3)
    points = []
    for i in range(coords.shape[0]):
        tmpX = int(np.round(f * coords[i, 0] / coords[i, 2] + px))
        tmpY = int(np.round(f * coords[i, 1] / coords[i, 2] + py))
        points.append(tuple((tmpX, tmpY)))
    print(points)
    box = drawLines(points, img)
    return box


def drawLines(points, img):
    box = np.copy(img)
    cv.line(box, points[0], points[1], (0, 255, 0), 2)
    cv.line(box, points[0], points[2], (0, 255, 0), 2)
    cv.line(box, points[0], points[4], (0, 255, 0), 2)
    cv.line(box, points[1], points[3], (0, 255, 0), 2)
    cv.line(box, points[1], points[5], (0, 255, 0), 2)
    cv.line(box, points[2], points[3], (0, 255, 0), 2)
    cv.line(box, points[2], points[6], (0, 255, 0), 2)
    cv.line(box, points[3], points[7], (0, 255, 0), 2)
    cv.line(box, points[4], points[5], (0, 255, 0), 2)
    cv.line(box, points[4], points[6], (0, 255, 0), 2)
    cv.line(box, points[5], points[7], (0, 255, 0), 2)
    cv.line(box, points[6], points[7], (0, 255, 0), 2)
    return box


def find_points(depth, img, threshold, x1=x1l, x2=x2l, y1=y1l, y2=y2l):
    patch = img[y1:y2 + 1, x1:x2 + 1]
    depth_patch = depth[y1:y2 + 1, x1:x2 + 1]
    max_i = depth_patch.max()
    min_i = depth_patch.min()
    threshold1 = ((max_i + min_i) * 1./2) * 0.7
    threshold2 = ((max_i + min_i) * 1./2) * 1.5
    new_img = np.zeros((depth_patch.shape[0], depth_patch.shape[1], 3))
    where_far = np.where(depth_patch > threshold2)
    where_near = np.where(depth_patch < threshold1)
    where_car = np.where((threshold1 < depth_patch) &
                         (depth_patch < threshold2))
    new_img[where_car] = np.array([0, 0, 255])
    new_img[where_far] = np.array([0, 255, 0])
    new_img[where_near] = np.array([255, 0, 0])
    save_image('./depth_patch.jpg', depth_patch)
    save_image('./new_img.jpg', new_img)


def find_cor_pixels(left_x1=x1l, left_y1=y1l, left_x2=x2l, left_y2=y2l,
                    img1=left_color, img2=right_color, patch_size=14):
    height, width = img1.shape[0], img1.shape[1]
    right_y1 = left_y1
    right_y2 = left_y2
    target_patch1 = img1[left_y1 - patch_size: left_y1 +
                         patch_size + 1, left_x1 - patch_size: left_x1 + patch_size + 1]
    target_patch2 = img1[left_y2 - patch_size: left_y2 +
                         patch_size + 1, left_x2 - patch_size: left_x2 + patch_size + 1]
    best_score1 = None
    best_ind1 = None
    best_score2 = None
    best_ind2 = None
    for x in range(patch_size, width - patch_size):
        source_patch1 = img2[right_y1 - patch_size: right_y1 +
                             patch_size + 1, x - patch_size: x + patch_size + 1]
        source_patch2 = img2[right_y2 - patch_size: right_y2 +
                             patch_size + 1, x - patch_size: x + patch_size + 1]
        score1 = ssd(source_patch1, target_patch1)
        if best_score1 is None or score1 < best_score1:
            best_score1 = score1
            best_ind1 = x
        score2 = ssd(source_patch2, target_patch2)
        if best_score2 is None or score2 < best_score2:
            best_score2 = score2
            best_ind2 = x
    x1, x2 = best_ind1 - patch_size, best_ind2
    y1, y2 = right_y1, right_y2
    img1 = draw_box(left_x1, left_y1, left_x2, left_y2, img1)
    img2 = draw_box(x1, y1, x2, y2, img2)
    img = np.vstack((img1, img2))
    print(x1, y1, x2, y2)
    cv.imwrite('./stack.jpg', img)
    return x1, y1, x2, y2


# def get_patch_depth(x1l, y1l, x2l, y2l, x1r, y1r, x2r, y2r, img1=left_color, img2=right_color, f=f, T=baseline, patch_size=14):
#     car_height, car_width = abs(y2l - y1l + 1), abs(x2l - x1l + 1)
#     diff = np.zeros((car_height, car_width))
#     scores = np.zeros((car_height, car_width))
#     ite_num = car_height * car_width * (img1.shape[1] - 2 * patch_size)
#     bar = progressbar.ProgressBar(maxval=ite_num,
#                                   widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
#     bar.start()
#     count = 0
#     # img = np.hstack((img1, car_depth))
#     for y1 in range(y1l, y2l + 1):
#         y2 = y1
#         for x1 in range(x1l, x2l + 1):
#             target_patch = img1[y1 - patch_size: y1 +
#                                 patch_size + 1, x1 - patch_size: x1 + patch_size + 1]
#             best_score = None
#             best_ind = None
#             for x2 in range(patch_size, img1.shape[1] - patch_size):
#                 source_patch = img2[y2 - patch_size: y2 +
#                                     patch_size + 1, x2 - patch_size: x2 + patch_size + 1]
#                 score = ssd(target_patch, source_patch)
#                 count += 1
#                 bar.update(count)
#                 if best_score is None or score < best_score:
#                     best_score = score
#                     best_ind = x2

#             diff[y1 - y1l, x1 - x1l] = abs(best_ind - x1)
#             scores[y1 - y1l, x1 - x1l] = abs(best_score)
#     scores = np.interp(scores, (scores.min(), scores.max()), (0, 1.0))
#     diff += scores
#     print(f'max dif is {diff.max()}')
#     print(f'min dif is {diff.min()}')
#     car_depth = np.divide(f * T, diff)
#     car_depth = np.interp(
#         car_depth, (car_depth.min(), car_depth.max()), (0, 255.0))

#     max_y_ind = None
#     max_y_diff = None
#     second_max_y_ind = None
#     second_max_y_diff = None
#     x = 0
#     for y in range(car_height - 1):
#         diff = abs(car_depth[y, 0] - car_depth(y + 1, 0))
#         if max_y_diff is None or diff > max_y_diff:

#     for x in range(car_width):

#     cv.imwrite('./car_depth.jpg', car_depth)


if __name__ == '__main__':
    # depth = load_grey_scale_image('./000020.png')

    # left_color = load_color_image('./A4_files/000020_left.jpg')
    # depth = np.interp(depth, (depth.min(), depth.max()), (0.0, 10))

    # x1r = math.floor(f * T / depth[y1l, x1l] + px)
    # x2r = math.floor(f * T / depth[y2l, x1l] + px)
    # y1r = y1l
    # y2r = y2l
    # new_img = draw_box(x1r, y1r, x2r, y2r, left_color)
    # save_image('./out.jpg', new_img)
    diff = scan_all()
    depth = calculate_depth(diff)
    # img = draw_3D_box(depth, left_color, 6)
    # cv.imwrite('./3d.jpg', img)
    # find_points(depth, left_color, 10)
    x1r, y1r, x2r, y2r = find_cor_pixels()
    # get_patch_depth(x1l, y1l, x2l, y2l, x1r, y1r, x2r, y2r)
    calculate_3d(depth, left_color, x1l, y1l, x2l, y2l, x1r, y1r, x2r, y2r)
