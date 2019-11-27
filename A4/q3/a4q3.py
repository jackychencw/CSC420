import cv2
import numpy as np
from skimage.draw import line_aa
from operator import itemgetter


def sift(img, nfeatures=10, ct=0.04, et=5, sigma=20):
    print("Finding Key Points ... ")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(
        contrastThreshold=ct, edgeThreshold=et, sigma=sigma)
    kp, des = sift.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(
        gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print("Done finding key points")
    return kp, des, img


def sift_color(img, nfeatures=100, ct=0.04, et=10, sigma=5):
    print("Finding Key Points ... ")
    sift = cv2.xfeatures2d.SIFT_create(
        contrastThreshold=ct, edgeThreshold=et, sigma=sigma)
    kp, des = sift.detectAndCompute(img, None)
    img = cv2.drawKeypoints(
        img, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print("Done finding key points")
    return kp, des, img


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
    y = img.shape[0]
    x = img.shape[1]
    r0 = max(r - l, 0)
    r1 = min(r + l + 1, y)
    c0 = max(c - l, 0)
    c1 = min(c + l + 1, x)
    img[r0: r1, c0: c1] = color
    return img


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


def fm(img1, img2):
    sf = cv.SIFT()
    kp1, des1 = sf.detectAndCompute(img1, None)
    kp2, des2 = sf.detectAndCompute(img2, None)

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
            pts2.append(kp2[m, trainIdx].pt)
            pts1.append(kp1[m, queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundementalMat(pts1, pts2, cv2.FM_LMEDS)

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    return pts1, pts2


if __name__ == '__main__':
    # img1_path = './images/img1.jpeg'
    # img1_rotate_path = './images/img1_rotate.jpeg'
    # img1_moved_path = './images/img1_moved.jpeg'

    # img1_origin = cv2.imread(img1_path)
    # img1_rotate = cv2.imread(img1_rotate_path)
    # img1_moved = cv2.imread(img1_moved_path)

    # img2_path = './images/img2.jpeg'
    # img2_rotate_path = './images/img2_rotate.jpeg'
    # img2_moved_path = './images/img2_moved.jpeg'

    # img2_origin = cv2.imread(img2_path)
    # img2_rotate = cv2.imread(img2_rotate_path)
    # img2_moved = cv2.imread(img2_moved_path)

    img3_path = './images/img3.jpeg'
    img3_rotate_path = './images/img3_rotate.jpeg'
    img3_moved_path = './images/img3_moved.jpeg'

    img3_origin = cv2.imread(img3_path)
    img3_rotate = cv2.imread(img3_rotate_path)
    img3_moved = cv2.imread(img3_moved_path)

    # (a)
    # new_img = match_images(img3_origin, img3_rotate)
    # cv2.imwrite('./results/img3_match_origin_rotate.jpeg', new_img)

    # (b)
