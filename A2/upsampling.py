import numpy as np
import cv2
DATA_PATH = "./CSC420-A2/"
OUTPUT_PATH = "./Output/"

def load_color_image(filename):
    image = cv2.imread(DATA_PATH + filename)
    return image

def load_grey_scale_image(filename):
    image = cv2.imread(DATA_PATH + filename, 0)
    return image

def save_image(filename, img, path=OUTPUT_PATH):
    cv2.imwrite(path + filename, img)
    print "Image {} saved.".format(filename)

def linear_interpolation(image, upsampling_scale, axis):
    y = image.shape[0]
    x = image.shape[1]
    z = image.shape[2]
    # Linear interpolation on x axis
    if axis == 0:
        result = np.zeros((y, (x-1) * upsampling_scale + 1, z))
        y_r = result.shape[0]
        x_r = result.shape[1]
        for i in range(y_r):
            for j in range(x_r - upsampling_scale + 1):
                if j % upsampling_scale == 0:
                    result[i,j] = image[i, j/upsampling_scale]
                else:
                    residual = j%upsampling_scale
                    a = image[i, (j - residual)/upsampling_scale]
                    b = image[i, (j + upsampling_scale - residual)/upsampling_scale]
                    result[i,j] = ((upsampling_scale - residual) * 1.0 /upsampling_scale) * a \
                    + (residual * 1.0/upsampling_scale) * b
    # Linear interpolation on y axis
    elif axis == 1:
        result = np.zeros(((y-1) * upsampling_scale + 1, x, z))
        y_r = result.shape[0]
        x_r = result.shape[1]
        for i in range(y_r):
            for j in range(x_r):
                if i % upsampling_scale == 0:
                    result[i,j] = image[i/upsampling_scale, j]
                else:
                    residual = j%upsampling_scale
                    a = image[(i - residual)/upsampling_scale, j]
                    b = image[(i + upsampling_scale - residual)/upsampling_scale, j]
                    result[i,j] = ((upsampling_scale - residual) * 1.0 /upsampling_scale) * a \
                    + (residual * 1.0/upsampling_scale) * b
    return result

def upsampling(image, upsampling_time):
    filter_size = upsampling_time + 1


if __name__ == "__main__":
    image = load_color_image("bee.jpg")
    new_image_1 = linear_interpolation(image, 4, 0)
    new_image_2 = linear_interpolation(new_image_1, 4, 1)
    save_image("upsampled.jpg", new_image_2)