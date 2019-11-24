from utils import *
import numpy as np

def ssd(patch1, patch2):
    diff = patch1 - patch2
    ssd = np.sum(diff**2)
    return ssd

def nc(patch1, patch2):
    a = patch1 * patch2
    b = np.sum(patch1 ** 2) * np.sum(patch2 ** 2)
    c = a * 1./b
    return c
