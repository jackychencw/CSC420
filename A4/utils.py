import cv2 as cv

def load_color_image(filepath):
    image = cv.imread(filepath)
    return image

def load_grey_scale_image(filepath):
    image = cv.imread(filepath, 0)
    return image

def show_image(img):
    cv.imshow("Showing image",img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def read_file(fname):
    fo = open(fname)
    lines = fo.readlines()
    fo.close()
    return lines