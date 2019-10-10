import matplotlib.pyplot as plt
import cv2
import numpy as np

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

def sift(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img, None)
    img = cv2.drawKeypoints(gray, kp, img)
    return img

if __name__ == "__main__":
    img = load_color_image("sample1.jpg")
    result = sift(img)
    cv2.imwrite('sift_keypoints.jpg',result)
    