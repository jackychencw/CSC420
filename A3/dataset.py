from utils import *
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
import numpy as np
import cv2 as cv


class CatDataset(Dataset):
    def __init__(self, rootdirectory):
        input_directory = rootdirectory + 'input/'
        mask_directory = rootdirectory + 'mask/'
        self.len = len(os.listdir(input_directory))
        self.X = torch.empty((self.len, 128, 128, 3))
        self.Y = torch.empty((self.len, 128, 128))
        for inputfilename in os.listdir(input_directory):
            input_img = load_color_image(
                os.path.join(input_directory, inputfilename))
            if input_img is not None:
                input_img = cv.resize(input_img, (int(IMAGE_H), int(IMAGE_W)))
                new_input_img = torch.from_numpy(input_img)
                self.X.add(new_input_img)
        for maskfilename in os.listdir(mask_directory):
            mask_img = load_grey_scale_image(
                os.path.join(mask_directory, maskfilename))
            if mask_img is not None:
                mask_img = cv.resize(mask_img, (int(IMAGE_H), int(IMAGE_W)))
                new_mask_img = torch.from_numpy(mask_img)
                self.Y.add(new_mask_img)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])
