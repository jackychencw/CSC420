import numpy as np
import cv2
DATA_PATH = "./CSC420-A2/"
OUTPUT_PATH = "./Output/1/"

def load_color_image(filename):
    image = cv2.imread(DATA_PATH + filename)
    return image

def load_grey_scale_image(filename):
    image = cv2.imread(DATA_PATH + filename, 0)
    return image

def save_image(filename, img, path=OUTPUT_PATH):
    cv2.imwrite(path + filename, img)
    print("Image {} saved.".format(filename))

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

def bilinear_interpolation(image, upsampling_scale):
    y, x, z = image.shape[0], image.shape[1], image.shape[2]
    m = np.zeros(((y-1) * upsampling_scale + 1, (x-1) * upsampling_scale \
    + 1, z))
    for i in range(y):
        for j in range(x):
            m[i * upsampling_scale, j*upsampling_scale] = image[i,j]
    filter_size = 2 * upsampling_scale - 1
    f = np.zeros((filter_size,filter_size))
    for i in range(filter_size):
        for j in range(filter_size):
            if i > filter_size / 2:
                y = filter_size - i
            else:
                y = i + 1
            if j > filter_size / 2:
                x = filter_size - j 
            else:
                x = j + 1
            f[i,j] = (y * x * 1.0)/(upsampling_scale * upsampling_scale)
    print f
    m = MyConvolution(m, f, "same")
    return m

def MyConvolution(img, h, mode):
    img_x = img.shape[1]
    img_y = img.shape[0]
    h_x = h.shape[1]
    h_y = h.shape[0]
    l = h_x//2
    h = np.flip(np.flip(h, 0),1)
    f = h.flatten('F')
    
    if mode == "full":
        new_x = img_x + (h_x - 1) * 2
        new_y = img_y + (h_y - 1) * 2
        
        new_img = np.zeros((new_y, new_x))
        
        padding_img = np.pad(img, ((h_y - 1, h_y - 1), (h_x - 1, h_x - 1)), "constant")
        
        for i in range(l, new_y - l):
            for j in range(l, new_x - l):
                t = padding_img[i - l: i + l + 1, j - l : j + l + 1]
                t = t.flatten('F')
                result = np.dot(f.T, t)
                new_img[i,j] = result
                
    elif mode == "same":
        new_img = np.zeros(img.shape)
        half_y = h_y//2
        half_x = h_x//2
        padding_img = np.pad(img, ((half_y, half_y),(half_x, half_x),(0,0)), "constant")
        
        for i in range(l, img_y - l + 1):
            for j in range(l, img_x - l + 1):
                t = padding_img[i - l: i + l + 1, j - l : j + l + 1]
                t = t.reshape((1,-1,3))
                result = np.dot(f.T, t)
                new_img[i,j] = result
                
    elif mode == "valid":
        new_x = img_x - h_x + 1
        new_y = img_y - h_y + 1
        new_img = np.zeros((new_y, new_x))
        
        for i in range(l, new_y - l + 1):
            for j in range(l, new_x - l + 1):
                t = img[i - l: i + l + 1, j - l : j + l + 1]
                t = t.flatten('F')
                result = np.dot(f.T, t)
                new_img[i,j] = result
    return new_img


def upsampling(image, upsampling_time):
    filter_size = upsampling_time + 1

def concatenate(i1, i2):
    assert i1.shape[0] == i2.shape[0]
    output = np.concatenate((i1, i2), axis = 1)
    return output

# Make image B has same rows as image A
def add_zeros(imgA, imgB):
    rowA = imgA.shape[0]
    rowB = imgB.shape[0]
    dif = abs(rowB - rowA)
    colB = imgB.shape[1]
    depB = imgB.shape[2]
    new_m = np.zeros((dif, colB, depB))
    output = np.concatenate((imgB, new_m), axis=0)
    return output


if __name__ == "__main__":
    # image = load_color_image("bee.jpg")
    # print(image.shape)
    # new_image_1 = linear_interpolation(image, 4, 0)
    # new_image_2 = linear_interpolation(new_image_1, 4, 1)
    # save_image("upsampled.jpg", new_image_2)

    # image = add_zeros(new_image_2, image)
    # print(new_image_2.shape)
    # print(image.shape)
    # output = concatenate(new_image_2, image)
    # save_image("concatenated.jpg", output)

    image = load_color_image("bee.jpg")
    new_image = bilinear_interpolation(image, 4)
    image = load_color_image("upsampled.jpg")
    print image.shape
    print new_image.shape
    save_image("new.jpg", new_image)
    dif = new_image - image
    dif = dif * 5
    save_image("dif.jpg", dif)
    new_image = concatenate(image, new_image)
    save_image("con.jpg", new_image)
    
    