import cv2 as cv
import numpy as np

def preprocess_dic(img):
    # CLAHE to increase variance of the distribution of pixels
    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    return img

def preprocess_fluo(img):
    img -= img.min()
    img *= (255 // img.max())

    msk = (img > img.mean())
    img *= msk

    return img

def preprocess_phc(img):
    # Background subtraction from assignment 1
    kernel = np.ones((15, 15),np.uint8)
    B = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    img = cv.subtract(img, B)

    img -= img.min()
    img *= (255 // img.max())

    msk = (img > img.mean())
    img *= msk

    return img
