import matplotlib.pyplot as plt
import cv2
import numpy as np
from operator import itemgetter
from skimage.draw import line_aa

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
def sift(img, nfeatures=100, ct = 0.04, et = 5, sigma = 20):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=ct, edgeThreshold = et, sigma = sigma)
    kp, des = sift.detectAndCompute(gray,None)
    img = cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return kp, des, img

def match(kp1, des1, kp2, des2, max_n = 10):
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
    dis = des1 - des2
    l2norm = np.linalg.norm(dis, 2)
    return l2norm

def concatenate(i1, i2):
    assert i1.shape == i2.shape
    output = np.concatenate((i1, i2), axis = 1)
    return output

if __name__ == "__main__":
    # Load sample1 and find keypoints, descriptors
    sample1 = load_color_image("sample1.jpg")
    kp1, des1, result1 = sift(sample1)
    save_image("sampl1_kp.jpg", result1)

     # Load sample2 and find keypoints, descriptors 
    sample2 = load_color_image("sample2.jpg")
    kp2, des2, result2 = sift(sample2)
    save_image("sampl2_kp.jpg", result2)

    # find a list of points that are closly matched between sample1 keypoints and sample2
    #  keypoints
    r = match(kp1, des1, kp2, des2)

    # Plot keypoints on both images
    a1 = load_color_image("sample1.jpg")
    a2 = load_color_image("sample2.jpg")
    gray1 = cv2.cvtColor(a1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(a2, cv2.COLOR_BGR2GRAY)
    test1 = cv2.drawKeypoints(a1, [i[0] for i in r], a1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    test2 = cv2.drawKeypoints(a2, [i[1] for i in r], a2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    test1_gray = cv2.drawKeypoints(gray1, [i[0] for i in r], a1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    test2_gray = cv2.drawKeypoints(gray2, [i[1] for i in r], a2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Concatenate two images
    test = concatenate(test1, test2)
    horizontal_shift = test1.shape[1]
    for i in r:
        r0 = int(i[0].pt[1])
        c0 = int(i[0].pt[0])
        r1 = int(i[1].pt[1])
        c1 = int(i[1].pt[0]) + horizontal_shift
        print(r0, c0, r1, c1)
        rr, cc, val = line_aa(r0, c0, r1, c1)
        print(rr)
        test[rr, cc] = np.array([0,0,255])
    save_image("test.jpg", test)

    test_gray = concatenate(test1_gray, test2_gray)
    save_image("test_gray.jpg", test_gray)

    