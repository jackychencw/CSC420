import os
import tensorflow as tf 
import cv2 as cv
from tensorflow.keras.preprocessing.image import save_img, load_img
def present(input_path, mask_path):
    # for inputfilename in os.listdir(input_path):
    #     input_img = load_img(
    #         os.path.join(input_path, inputfilename), color_mode="color")

    for maskfilename in os.listdir(mask_path):
        mask_img = cv.imread(os.path.join(mask_path, maskfilename), 0)
        edge = cv.Canny(mask_img, mask_img.shape[0], mask_img.shape[1])
        save_img("./edge.jpg", edge)
        return

if __name__ == "__main__":
    input_path = "./Output/x/"
    mask_path = "./Output/y/"
    present(input_path, mask_path)