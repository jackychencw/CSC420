import torch
import os
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import numpy as np
from skimage.transform import resize

class CatDataset(Dataset):
    def __init__(self, rootdirectory, im_height = 128, im_width = 128):
        input_directory = rootdirectory + 'input/'
        mask_directory = rootdirectory + 'mask/'
        self.len = len(os.listdir(input_directory))
        self.X = np.zeros((self.len, im_height, im_width, 1), dtype=np.float32)
        self.Y = np.zeros((self.len, im_height, im_width, 1), dtype=np.float32)

        id = 0
        for inputfilename in os.listdir(input_directory):
            input_img = load_img(
                os.path.join(input_directory, inputfilename), color_mode="grayscale")
            if input_img is not None:
                input_img = img_to_array(input_img)
                input_img = resize(input_img, (int(im_height), int(im_width)))
                new_input_img = torch.from_numpy(input_img)
                self.X[id, ..., 0] = new_input_img.squeeze() / 255
            id += 1
        id = 0
        for maskfilename in os.listdir(mask_directory):
            mask_img = load_img(
                os.path.join(mask_directory, maskfilename), color_mode="grayscale")
            if mask_img is not None:
                mask_img = img_to_array(mask_img)
                mask_img = resize(mask_img, (int(im_height), int(im_width)))
                new_mask_img = torch.from_numpy(mask_img)
                self.Y[id] = new_mask_img / 255
            id += 1

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])
