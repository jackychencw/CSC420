import matplotlib.pyplot as plt
import cv2
import numpy as np
from operator import itemgetter
from skimage.draw import line_aa

sift = cv2.xfeatures2d.SIFT_create()


def feature_matching(im1, im2, max_n=8):
    img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # find the keypoints and descriptors with SIFT
    sift = cv2.xfeatures2d.SIFT_create(2000)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []
    figsize = (15, 15)
    fig = plt.figure(figsize=figsize)
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    # random.shuffle(good)
    pts1 = np.int32(pts1)[:max_n]
    pts2 = np.int32(pts2)[:max_n]
    img3 = cv2.drawMatches(im1, kp1, im2, kp2, good[:max_n], None, flags=4)

    return img3, pts1, pts2


def fundamental(points1, points2):
    width = points1.shape[1]
    m = np.zeros((width, 9))
    for n in range(width):
        xrn = points1[0, n]
        xln = points2[0, n]
        yrn = points1[1, n]
        yln = points2[1, n]
        m[n] = np.array([xrn * xln, xrn * yln, xrn, yrn *
                         xln, yrn * yln, yrn, xln, yln, 1])
    U, S, V = np.linalg.svd(m)
    F = V[-1].reshape(3, 3)
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))
    return F/F[2, 2]


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def epipolar(img1, img2):
    print("finding kps")
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    print("done")

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)[:8]
    pts2 = np.int32(pts2)[:8]
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    # We select only inlier points

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    return img5, img3


def rectify(img1, img2, f=True):
    print("rectifying")
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    if f:
        F = fundamental(pts1, pts2)
    else:
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
    res, H1, H2 = cv2.stereoRectifyUncalibrated(
        pts1, pts2, F, img1.shape[:2], threshold=10)
    left = cv2.warpPerspective(img1, H1, img1.shape[:2])
    right = cv2.warpPerspective(img2, H2, img2.shape[:2])
    print("done")
    return left, right


if __name__ == '__main__':

    img1_path = './images/img1.jpeg'
    img2_path = './images/img2.jpeg'
    img3_path = './images/img3.jpeg'

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img3 = cv2.imread(img3_path)

    img1_gray = cv2.imread(img1_path, 0)
    img2_gray = cv2.imread(img2_path, 0)
    img3_gray = cv2.imread(img3_path, 0)

    # (a)
    # i1i2, pts1_i1i2, pts2_i1i2 = feature_matching(
    #     img1, img2)
    # cv2.imwrite('./results/a_i1i2.jpeg', i1i2)

    # i1i3, pts1_i1i3, pts2_i1i3 = feature_matching(
    #     img1, img3)
    # cv2.imwrite('./results/a_i1i3.jpeg', i1i3)

    # (b)
    # f_i1i2 = fundamental(pts1_i1i2, pts2_i1i2)
    # print(f_i1i2)
    # f_i1i3 = fundamental(pts1_i1i3, pts2_i1i3)
    # print(f_i1i3)

    # (c)

    left_i1i2, right_i1i2 = epipolar(img1_gray, img2_gray)
    cv2.imwrite('./results/c_left_i1i2.jpeg', left_i1i2)
    cv2.imwrite('./results/c_right_i1i2.jpeg', right_i1i2)

    left_i1i3, right_i1i3 = epipolar(img1_gray, img3_gray)
    cv2.imwrite('./results/c_left_i1i3.jpeg', left_i1i3)
    cv2.imwrite('./results/c_right_i1i3.jpeg', right_i1i3)

    # (d)
    left_i1i2 = cv2.imread('./results/c_left_i1i2.jpeg')
    right_i1i2 = cv2.imread('./results/c_right_i1i2.jpeg')
    l_i1i2, r_i1i2 = rectify(left_i1i2, right_i1i2, f=True)
    new = np.hstack((l_i1i2, r_i1i2))
    cv2.imwrite('./results/d_r_i1i2.jpeg', r_i1i2)
    cv2.imwrite('./results/d_l_i1i2.jpeg', l_i1i2)
    cv2.imwrite('./results/d_new_i1i2.jpeg', new)

    left_i1i3 = cv2.imread('./results/c_left_i1i3.jpeg')
    right_i1i3 = cv2.imread('./results/c_right_i1i3.jpeg')
    l_i1i3, r_i1i3 = rectify(left_i1i3, right_i1i3, f=True)
    cv2.imwrite('./results/d_r_i1i3.jpeg', r_i1i3)
    cv2.imwrite('./results/d_l_i1i3.jpeg', l_i1i3)
    new = np.hstack((l_i1i3, r_i1i3))
    cv2.imwrite('./results/d_new_i1i3.jpeg', new)

    # (e)
    left_i1i2 = cv2.imread('./results/c_left_i1i2.jpeg')
    right_i1i2 = cv2.imread('./results/c_right_i1i2.jpeg')
    l_i1i2, r_i1i2 = rectify(left_i1i2, right_i1i2, f=False)
    cv2.imwrite('./results/e_r_i1i2.jpeg', r_i1i2)
    cv2.imwrite('./results/e_l_i1i2.jpeg', l_i1i2)
    new = np.hstack((l_i1i2, r_i1i2))
    cv2.imwrite('./results/e_new_i1i2.jpeg', new)

    left_i1i3 = cv2.imread('./results/c_left_i1i3.jpeg')
    right_i1i3 = cv2.imread('./results/c_right_i1i3.jpeg')
    l_i1i3, r_i1i3 = rectify(left_i1i3, right_i1i3, f=False)
    cv2.imwrite('./results/e_r_i1i3.jpeg', r_i1i3)
    cv2.imwrite('./results/e_l_i1i3.jpeg', l_i1i3)
    new = np.hstack((l_i1i3, r_i1i3))
    cv2.imwrite('./results/e_new_i1i3.jpeg', new)
