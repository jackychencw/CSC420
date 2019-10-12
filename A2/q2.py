import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_laplace, minimum_filter
import math
import scipy

DATA_PATH = "./CSC420-A2/"
OUTPUT_PATH = "./Output/2/"

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

def harris_corner_detection_lambda(img, alpha, threshold):
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
    lambda1 = Ix2_blur
    lambda2 = Iy2_blur
    y = lambda1.shape[0]
    x = lambda1.shape[1]
    where_corner = zip(*np.where((lambda1 > threshold) & (lambda2 > threshold)))
    R = lambda1 + lambda2
    for corner in where_corner:
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

def brown_corner_detection(img):
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
    corner_list = zip(*np.where(hm > 1000000))
    for corner in corner_list:
        filter_size = 10
        i = corner[0]
        j = corner[1]
        i_s = max(i - filter_size, 0)
        i_d = min(y - 1, i + filter_size)
        j_s = max(j - filter_size, 0)
        j_d = min(x - 1, j + filter_size)
        matrix = hm[i_s:i_d, j_s:j_d]
        if matrix.max() == hm[i,j]:
            dot_d = 3
            i_s = max(i - dot_d, 0)
            i_d = min(y - 1, i + dot_d)
            j_s = max(j - dot_d, 0)
            j_d = min(x - 1, j + dot_d)
            img[i_s:i_d,j_s:j_d] = np.array([0,0,255])
    return img

def blob_detection(img, sigmas=[1,2,4,8,16,32,64,128]):
    scales = [5,10,20, 30,40, 80, 160]
    y, x = img.shape[0], img.shape[1]
    potentials = []
    img = cv2.GaussianBlur(img,(5,5),7)
    LoG = gaussian_laplace(1.0 * img, sigmas[0])
    threshold = 0.6 * np.max(LoG)
    points = zip(*np.where(LoG[:,:] > threshold))
    print len(points)
    for point in points:
        i, j = point[0], point[1]
        extreme = None
        for scale in scales:
            r0 = max(0, i - scale)
            r1 = min(i + scale, y - 1)
            c0 = max(0, j - scale)
            c1 = min(j + scale, x - 1)
            sub_image = img[r0:r1,c0:c1]
            patch = gaussian_laplace(1.0 * sub_image, sigmas[0])
            zc = np.max(patch) > 0 and np.min(patch) < 0
            if zc and (patch[scale,scale] == np.max(patch)):
                if extreme is None:
                    extreme = (i, j, scale, patch[scale,scale])
                elif patch[scale, scale] > extreme[3]:
                    extreme = (i, j, scale, patch[scale,scale])
            if extreme is not None:
                potentials.append(extreme)
    return potentials






# def laplacian_gaussian(patch, sigma):
#     LoG = scipy.ndimage.gaussian_laplace(patch, sigma)
#     f = scipy.ndimage.gaussian_filter(np.ones(patch.shape), sigma)
#     print f
    
#     # LoG = patch * LoG_filter
#     return LoG
    



def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

# def draw_corners(img, corners):
#     y,x = img.shape[0], img.shape[1]
#     for corner in corners:
#         i,j,d = corner[0],corner[1],5
#         r0 = max(0,i - d)
#         r1 = min(y-1, i+d)
#         c0 = max(0,j-d)
#         c1 = min(x-1, j+d)
#         img[r0:r1,c0:c1] = np.array([0,0,255])
#     return img

def draw_circles(img, kps):
    for kp in kps:
        y = kp[0]
        x = kp[1]
        r = kp[2]
        cv2.circle(img, (int(x),int(y)), int(r), (0,0,255))
    return img


if __name__ == "__main__":
    img = load_gray_scale_image("building.jpg")
    
    color_img = load_color_image("building.jpg")
    # new_img = draw_circles(color_img, blobs_log)
    # save_image("circles2.jpg", new_img)
    LoG = blob_detection(img)
    new_img = draw_circles(color_img, LoG)
    save_image("LoG.jpg", new_img)

    # threshold = 10**6
    # haris1 = harris_corner_detection_lambda(img, 0.04, threshold)
    # haris2 = harris_corner_detection(img, 0.05, threshold)
    # haris3 = harris_corner_detection(img, 0.06, threshold)
    # save_image("harris1_lambda.jpg", haris1)
    # save_image("harris2_s1.jpg", haris2)
    # save_image("harris3_s1.jpg", haris3)
    # brown = brown_corner_detection(img)
    # save_image("brown.jpg", brown)
    # rotate_60 = rotateImage(img, 60)
    # harris_rotate_60 = harris_corner_detection(rotate_60, 0.06, threshold)
    # save_image("harris rotate 60.jpg", harris_rotate_60)
