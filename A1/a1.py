import numpy as np
import cv2
import scipy.misc
from scipy.ndimage.filters import gaussian_filter
path = "/Users/jackychencw/Desktop/CSC420/A1/"

def readImg(filename):
    image = cv2.imread(path + filename, 0)
    return image

# 4.(a)
def MyCorrelation(I, h, mode):
    img = readImg(I)
    img_x = img.shape[1]
    img_y = img.shape[0]
    h_x = h.shape[1]
    h_y = h.shape[0]
    l = h_x//2
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
        padding_img = np.pad(img, ((half_y, half_y),(half_x, half_x)), "constant")
        
        for i in range(l, img_y - l + 1):
            for j in range(l, img_x - l + 1):
                t = padding_img[i - l: i + l + 1, j - l : j + l + 1]
                t = t.flatten('F')
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

# 4.(b)
def MyConvolution(I, h, mode):
    img = readImg(I)
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
        padding_img = np.pad(img, ((half_y, half_y),(half_x, half_x)), "constant")
        
        for i in range(l, img_y - l + 1):
            for j in range(l, img_x - l + 1):
                t = padding_img[i - l: i + l + 1, j - l : j + l + 1]
                t = t.flatten('F')
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

# 4.(c)
def portrait(I, h, mode, mask):
    img = readImg(I)
    new_img =gaussian_filter(img, sigma=7)
    result = np.zeros(new_img.shape)
    where_0 = np.where(mask == 0)
    where_1 = np.where(mask == 1)
    result[where_0] = new_img[where_0]
    result[where_1] = img[where_1]
    return result

mask = np.zeros((200,200))
h = np.ones((3,3))
mask = np.pad(mask, ((300,300),(300,300)), "constant")
img = portrait("gray.jpg", h, "same", mask)
scipy.misc.imsave(path + 'portrait.jpg', img)

# 5.(b)
def isSeperable(m):
    u, s, vh = np.linalg.svd(m)
    vs = np.diag(s)
    threshold = 1e-7
    non_vanishing_sv = vs[vs > threshold]
    if len(non_vanishing_sv) == 1:
        vf = np.sqrt(s[0]) * u[:,0]
        hf = np.sqrt(s[0]) * vh[0]
        return True, vf, hf
    else:
        return False, None, None

# 6.(a)
def addRandNoise(I,m):
    image = cv2.imread(path + I)
    noise = np.random.uniform(-m, m, image.shape)
    rescaled_image = np.interp(image, (image.min(), image.max()), (0.0, 1.0))
    rescaled = rescaled_image + noise
    noised_image = np.interp(rescaled, (rescaled.min(), rescaled.max()), (0, 255.0))
    return noised_image

# 6.(b)
def denoise(I):
    image = cv2.imread(path + I)
    image = gaussian_filter(image, sigma=1)
    return image

# 6.(c)
def addSaltAndPepperNoise(I,density):
    image = cv2.imread(path + I, 0)
    print image
    noise = np.random.choice([0,1,2], image.shape[0] * image.shape[1], p = [1 - density, density/2, density/2])
    noise = noise.reshape(image.shape)
    where_salt = np.where(noise == 1)
    where_pepper = np.where(noise == 2)
    image[where_salt] = 255
    image[where_pepper] = 0
    return image

# 6.(d)
def denoise_sap(I):
    shift = 2
    filter_size = (2*shift+1)**2
    h = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,0,1,1],[1,1,0,1,1],[1,1,0,1,1]])
    hf = h.flatten('F')
    image = cv2.imread(path + I, 0)
    possible_noise = np.argwhere((image > 220) | (image < 35))
    possible_noise += shift
    image = np.pad(image,((shift,shift),(shift,shift)),'constant')
    for i in possible_noise:
        y = i[0]
        x = i[1]
        t = image[y - shift : y + shift + 1, x - shift : x + shift + 1]
        tf = t.flatten('F')
        r = np.dot(hf.T, tf)/( filter_size - 1)
        image[y,x] = r
    return image

# 6.(e)
def denoise(image):
    shift = 2
    filter_size = (2*shift+1)**2
    h = np.array([[1,1,1,1,1],[1,1,1,1,1],[1,1,0,1,1],[1,1,1,1,1],[1,1,1,1,1]])
    hf = h.flatten('F')
    
    possible_noise = np.argwhere((image == 255) | (image == 0))
    possible_noise += shift
    image = np.pad(image,((shift,shift),(shift,shift)),'constant')
    for i in possible_noise:
        y = i[0]
        x = i[1]
        t = image[y - shift : y + shift + 1, x - shift : x + shift + 1]
        tf = t.flatten('F')
        r = np.dot(hf.T, tf.T)/( filter_size - 1)
        image[y,x] = r
    return image

def denoise_color(I):
    image = cv2.imread(path + I, cv2.COLOR_BGR2RGB)
    print image.shape
    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]
    denoised_r = denoise(r)
    denoised_g = denoise(g)
    denoised_b = denoise(b)
    new_image = np.dstack([denoised_r, denoised_g, denoised_b])
    print new_image.shape
    return new_image
