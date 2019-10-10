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
    print "Image {} saved.".format(filename)

def harris_corner_detection(img, alpha, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),7)

    # Compute Ix and Iy
    Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)

    # Compute Ix^2, Iy^2 and IxIy
    IxIy = np.multiply(Ix, Iy)
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)

    # Blur Ix^2, Iy^2 and IxIy
    Ix2_blur = cv2.GaussianBlur(Ix2,(5,5),10) 
    Iy2_blur = cv2.GaussianBlur(Iy2,(5,5),10) 
    IxIy_blur = cv2.GaussianBlur(IxIy,(5,5),10)

    # Compute r = det - alpha * trace^2
    det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur,IxIy_blur)
    trace = Ix2_blur + Iy2_blur
    R = det - alpha * np.multiply(trace,trace)

    # Threshold and non-maximum suppression
    y = R.shape[0]
    x = R.shape[1]
    corner_list = zip(*np.where(R > threshold))
    for corner in corner_list:
        filter_size = 10
        i = corner[0]
        j = corner[1]
        i_s = max(i - filter_size, 0)
        i_d = min(y - 1, i + filter_size)
        j_s = max(j - filter_size, 0)
        j_d = min(x - 1, j + filter_size)
        matrix = R[i_s:i_d, j_s:j_d]
        if matrix.max() == R[i,j]:
            dot_d = 3
            i_s = max(i - dot_d, 0)
            i_d = min(y - 1, i + dot_d)
            j_s = max(j - dot_d, 0)
            j_d = min(x - 1, j + dot_d)
            img[i_s:i_d,j_s:j_d] = np.array([0,0,255])
    return img

def brown_corner_detection(img, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),7)

    # Compute Ix and Iy
    Ix = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5)
    Iy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5)

    # Compute Ix^2, Iy^2 and IxIy
    IxIy = np.multiply(Ix, Iy)
    Ix2 = np.multiply(Ix, Ix)
    Iy2 = np.multiply(Iy, Iy)

    # Blur Ix^2, Iy^2 and IxIy
    Ix2_blur = cv2.GaussianBlur(Ix2,(5,5),10) 
    Iy2_blur = cv2.GaussianBlur(Iy2,(5,5),10) 
    IxIy_blur = cv2.GaussianBlur(IxIy,(5,5),10)

    # Compute r = det - alpha * trace^2
    det = np.multiply(Ix2_blur, Iy2_blur) - np.multiply(IxIy_blur,IxIy_blur)
    trace = Ix2_blur + Iy2_blur
    where_zeros = np.where(trace==0)
    det[where_zeros] = 0
    trace[where_zeros] = 1
    hm = np.divide(det, trace)
    # Threshold and non-maximum suppression
    y = hm.shape[0]
    x = hm.shape[1]
    corner_list = zip(*np.where(hm > threshold))
    for corner in corner_list:
        filter_size = 10
        i = corner[0]
        j = corner[1]
        i_s = max(i - filter_size, 0)
        i_d = min(y - 1, i + filter_size)
        j_s = max(j - filter_size, 0)
        j_d = min(x - 1, j + filter_size)
        matrix = R[i_s:i_d, j_s:j_d]
        if matrix.max() == R[i,j]:
            dot_d = 3
            i_s = max(i - dot_d, 0)
            i_d = min(y - 1, i + dot_d)
            j_s = max(j - dot_d, 0)
            j_d = min(x - 1, j + dot_d)
            img[i_s:i_d,j_s:j_d] = np.array([0,0,255])
    return img

def LoG():
    return
    
def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


if __name__ == "__main__":
    img = load_color_image("building.jpg")
    threshold = 10**12
    haris1 = harris_corner_detection(img, 0.04, threshold)
    haris2 = harris_corner_detection(img, 0.05, threshold)
    haris3 = harris_corner_detection(img, 0.06, threshold)
    save_image("harris1.jpg", haris1)
    save_image("harris2.jpg", haris2)
    save_image("harris3.jpg", haris3)
    brown = brown_corner_detection(img, threshold)
    save_image("brown.jpg", brown)
    rotate_60 = rotateImage(img, 60)
    harris_rotate_60 = harris_corner_detection(rotate_60, 0.06, threshold)
    save_image("harris rotate 60.jpg", harris_rotate_60)
