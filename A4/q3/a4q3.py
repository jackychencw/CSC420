import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_laplace, minimum_filter
import math
import scipy


def sift(img, nfeatures=10, ct=0.04, et=5, sigma=20):
    new_img = np.copy(img)
    print("Finding Key Points ... ")
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(
        contrastThreshold=ct, edgeThreshold=et, sigma=sigma)
    kp, des = sift.detectAndCompute(gray, None)
    new_img = cv2.drawKeypoints(
        gray, kp, new_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print("Done finding key points")
    return kp, des, new_img


def sift_color(img, nfeatures=100, ct=0.04, et=10, sigma=5):
    new_img = np.copy(img)
    print("Finding Key Points ... ")
    sift = cv2.xfeatures2d.SIFT_create(
        contrastThreshold=ct, edgeThreshold=et, sigma=sigma)
    kp, des = sift.detectAndCompute(new_img, None)
    new_img = cv2.drawKeypoints(
        new_img, kp, new_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print("Done finding key points")
    return kp, des, new_img


def distance(des1, des2):
    # L2 Norm
    dis = des1 - des2
    l2norm = np.linalg.norm(dis, 2)

    # L1 Norm
    l1norm = np.linalg.norm(dis, 1)

    # L3 Norm
    l3norm = np.linalg.norm(dis, 3)

    return l2norm


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


def concatenate(i1, i2):
    assert i1.shape[0] == i2.shape[0]
    output = np.concatenate((i1, i2), axis=1)
    return output


def draw_square(img, r, c, l, color):
    img_cp = np.copy(img)
    y = img_cp.shape[0]
    x = img_cp.shape[1]
    r0 = max(r - l, 0)
    r1 = min(r + l + 1, y)
    c0 = max(c - l, 0)
    c1 = min(c + l + 1, x)
    img_cp[r0: r1, c0: c1] = color
    return img_cp


def match_images(img1, img2, color=False):
    if color:
        kp1, des1, img1_sift = sift_color(img1)
        kp2, des2, img2_sift = sift_color(img2)
    else:
        kp1, des1, img1_sift = sift(img1)
        kp2, des2, img2_sift = sift(img2)
    match_points = match(kp1, des1, kp2, des2)
    con = concatenate(img1, img2)
    h_shift = img1_sift.shape[1]
    for point in match_points:
        r0 = int(point[0].pt[1])
        c0 = int(point[0].pt[0])
        r1 = int(point[1].pt[1])
        c1 = int(point[1].pt[0]) + h_shift
        new_img = draw_square(con, r0, c0, 10, np.array([0, 0, 255]))
        new_img = draw_square(con, r1, c1, 10, np.array([0, 0, 255]))
        rr, cc, val = line_aa(r0, c0, r1, c1)
        new_img[rr, cc] = np.array([0, 255, 0])
    return new_img


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    img = concatenate(img1, img2)
    return img


def fm(img1, img2):
    kp1, des1, img1_sift = sift(img1)
    kp2, des2, img1_sift = sift(img2)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorighm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundementalMat(pts1, pts2, cv2.FM_LMEDS)

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)

    return pts1, pts2, lines1


if __name__ == '__main__':
    img1_path = './images/img1.jpeg'
    img2_path = './images/img2.jpeg'
    img3_path = './images/img3.jpeg'

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img3 = cv2.imread(img3_path)

    # (a)
    i1i2 = match_images(img1, img2)
    cv2.imwrite('./results/i1i2.jpeg', i1i2)
    i1i3 = match_images(img1, img3)
    cv2.imwrite('./results/i1i3.jpeg', i1i3)
    # (b)
    # pts1, pts2, lines = fm(img3_origin, img3_moved)
    # img = drawlines(img3_origin, img3_moved, lines, pts1, pts2)
    # cv2.imwrite("./results/epipolar.jpeg", img)
