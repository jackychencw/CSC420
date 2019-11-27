import matplotlib.pyplot as plt
import cv2
import numpy as np
from operator import itemgetter
from skimage.draw import line_aa

# Used opencv SIFT module to find keypoints and descriptors
# Used skimage to draw lines between points

DATA_PATH = "./CSC420-A2/"
OUTPUT_PATH = "./Output/"


def load_color_image(filename):
    image = cv2.imread(DATA_PATH + filename)
    return image


def load_gray_scale_image(filename):
    image = cv2.imread(DATA_PATH + filename, 0)
    return image


def save_image(filename, img, path=OUTPUT_PATH):
    cv2.imwrite(path + filename, img)

# Tune sigma to filter of small scale keypoints, higher sigma => less kp
# Tune et to filter edge keypoints, smaller et => less kp
# Tune ct to filter weak features, higher ct => less kp
# nfeatures: number of best feature kps


def sift(img, nfeatures=100, ct=0.04, et=5, sigma=20):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(
        contrastThreshold=ct, edgeThreshold=et, sigma=sigma)
    kp, des = sift.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(
        gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp, des, img


def sift_gray(gray, img, nfeatures=100, ct=0.04, et=5, sigma=20):
    sift = cv2.xfeatures2d.SIFT_create(
        contrastThreshold=ct, edgeThreshold=et, sigma=sigma)
    kp, des = sift.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(
        gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp, des, img


def sift_color(img, nfeatures=100, ct=0.04, et=10, sigma=5):
    sift = cv2.xfeatures2d.SIFT_create(
        contrastThreshold=ct, edgeThreshold=et, sigma=sigma)
    kp, des = sift.detectAndCompute(img, None)
    img = cv2.drawKeypoints(
        img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp, des, img


def match(kp1, des1, kp2, des2, max_n=12):
    lkp1 = len(kp1)
    lkp2 = len(kp2)
    point_list = []
    for i in range(lkp1):
        min_dis = None
        for j in range(lkp2):
            d1 = des1[i]
            d2 = des2[j]
            dis = distance(d1, d2)
            if (min_dis == None) or (dis < min_dis[2]):
                min_dis = (kp1[i], kp2[j], dis)
        point_list.append(min_dis)
    point_list.sort(key=itemgetter(2))
    return point_list[:min(len(point_list), max_n)]


def distance(des1, des2):
    # L2 Norm
    dis = des1 - des2
    l2norm = np.linalg.norm(dis, 2)

    # L1 Norm
    l1norm = np.linalg.norm(dis, 1)

    # L3 Norm
    l3norm = np.linalg.norm(dis, 3)

    return l2norm


def concatenate(i1, i2):
    assert i1.shape[0] == i2.shape[0]
    output = np.concatenate((i1, i2), axis=1)
    return output


def draw_square(img, r, c, l, color):
    y = img.shape[0]
    x = img.shape[1]
    r0 = max(r - l, 0)
    r1 = min(r + l + 1, y)
    c0 = max(c - l, 0)
    c1 = min(c + l + 1, x)
    img[r0: r1, c0: c1] = color
    return img

# Make image B has same rows as image A


def add_zeros(imgA, imgB):
    rowA = imgA.shape[0]
    rowB = imgB.shape[0]
    dif = abs(rowB - rowA)
    colB = imgB.shape[1]
    depB = imgB.shape[2]
    new_m = np.zeros((dif, colB, depB))
    output = np.concatenate((imgB, new_m), axis=0)
    print(output.shape)
    return output


def addRandNoise(img):
    noise = np.random.normal(0, 0.08, img.shape)
    rescaled_image = np.interp(img, (img.min(), img.max()), (0.0, 1.0))
    noised_rescaled_image = rescaled_image + noise
    noised_image = np.interp(noised_rescaled_image, (noised_rescaled_image.min(),
                                                     noised_rescaled_image.max()), (0, 255.0))
    return noised_image.astype(int)


if __name__ == "__main__":
    # Load sample1 and find keypoints, descriptors
    sample1 = load_color_image("noised1.jpg")
    kp1, des1, result1 = sift(sample1)
    save_image("sampl1_noised_kp.jpg", result1)

    # Load sample2 and find keypoints, descriptors
    sample2 = load_color_image("noised2.jpg")
    kp2, des2, result2 = sift(sample2)
    save_image("sampl2_noised_kp.jpg", result2)

    # find a list of points that are closly matched between sample1 keypoints and sample2
    #  keypoints
    r = match(kp1, des1, kp2, des2)

    # Plot keypoints on both images
    a1 = load_gray_scale_image("noised1.jpg")
    a2 = load_gray_scale_image("noised2.jpg")
    test1 = cv2.drawKeypoints(
        a1, [i[0] for i in r], a1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    test2 = cv2.drawKeypoints(
        a2, [i[1] for i in r], a2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Concatenate two images
    test = concatenate(test1, test2)
    horizontal_shift = test1.shape[1]
    for i in r:
        r0 = int(i[0].pt[1])
        c0 = int(i[0].pt[0])
        r1 = int(i[1].pt[1])
        c1 = int(i[1].pt[0]) + horizontal_shift
        test = draw_square(test, r0, c0, 10, np.array([0, 0, 255]))
        test = draw_square(test, r1, c1, 10, np.array([0, 0, 255]))
        rr, cc, val = line_aa(r0, c0, r1, c1)
        test[rr, cc] = np.array([0, 255, 0])
    save_image("noised_gray_l2norm.jpg", test)
