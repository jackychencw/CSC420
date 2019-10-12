from matplotlib import pyplot as plt
from scipy import ndimage,misc
from skimage import feature
import numpy as np
import math, cv2

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

def detect_blobs(image):
	images_sigma = []
	sigma = [1, 2.2, 3.4, 4.0, 4.6, 5.2, 5.8, 6.4, 8.5, 11, 15]
	for count in range(11):
		filtered_image = image * laplacian_of_gaussian(image, sigma[count])
		#filtered_image = filtered_image * (scale**2)
		filtered_image = filtered_image
		images_sigma.append(filtered_image)
		#print "count"
	stacked_images = np.dstack(images_sigma)
	print stacked_images.shape
	# print stacked_images
	# print sigma
	lm = feature.peak_local_max(stacked_images, threshold_abs=0, footprint=np.ones((3, 3, 3)), threshold_rel=10, exclude_border=True)
	lm = lm.astype(np.float64)
	lm = np.array(lm)
	lm[:, 2] = (lm[:, 2]).astype(int)
	count = 0
	for x in lm[:,2]:
		lm[count][2] = sigma[int(x)%11]*math.sqrt(2)
		count+=1
	#print lm
	for points in lm:
		cv2.circle(image,(int(points[1]),int(points[0])), int(points[2]), (255,255,255), 1)
	return image
	
def laplacian_of_gaussian(image,sigma):
    f_image = ndimage.filters.gaussian_laplace(image, sigma, output=None, mode='reflect')
    return f_image

def read_image(image_path):
    image = cv2.imread(image_path)
    return image

def display_image(image, title):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Read Image
    image = load_grey_scale_image('building.jpg')
    # display_image(image, "Blob")
    r = detect_blobs(image)
    save_image("log.jpg", r)

