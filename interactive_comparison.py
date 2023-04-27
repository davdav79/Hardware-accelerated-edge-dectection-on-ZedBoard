# TODO: 4 interactive windows 1st: original, 2nd preprocessed, 3rd sobel, 4th canny, 5th XOR

import cv2
import numpy as np
import preprocessing as pp


# TODO: original image:
# change images with slider (more or less noise)

# window for original image
cv2.namedWindow('Originalbild')

def image_change(x):
    # load image with noise
    global img
    img = cv2.imread(f"sample_images/Picture_Crossing_noise_{x*10}_pixelCnt_128_featureCnt_5.bmp")
    # show image with noise
    cv2.imshow('Originalbild', img)

# add slider for changing image with noise
cv2.createTrackbar('Noise', 'Originalbild', 0, 10, image_change)

# load default image
img = cv2.imread("sample_images/Picture_Crossing_noise_0_pixelCnt_128_featureCnt_5.bmp")
cv2.imshow('Originalbild', img)


# TODO: preprocessed image:
# slider for sigma
# slider for kernel_size_blur
# slider for different preprocessing methods (n is dependant on kernel size)
# slider for kernel size

# window for preprocessed image
cv2.namedWindow('vorverarbeitetes Bild')

# define parameters
sigma = 1
kernel_size_blur = 3
preprocessing_algorithm = 0
kernel_size = 3

# functions for preprocessing
def sigma(x):
    # define global variables
    global sigma
    global img
    global kernel_size_blur
    global preprocessing_algorithm
    global kernel_size
    # set sigma
    sigma = x
    # apply preprocessing
    # TODO: implement function in preprocessing.py
    img = pp.preprocessing(img, sigma, kernel_size_blur, preprocessing_algorithm, kernel_size)
    # show image
    cv2.imshow('vorverarbeitetes Bild', img)


def kernel_size_blur(x):
    # define global variables
    global kernel_size_blur
    # set kernel size
    kernel_size_blur = 2 * x + 3

def preprocessing_algorithm(x):
    # define global variables
    global preprocessing_algorithm
    # set preprocessing algorithm
    preprocessing_algorithm = x

def kernel_size(x):
    # define global variables
    global kernel_size
    # set kernel size
    kernel_size = 2 * x + 3

# add sliders for preprocessing
cv2.createTrackbar('sigma', 'vorverarbeitetes Bild', 0, 10, sigma)
cv2.createTrackbar('kernel_size_blur', 'vorverarbeitetes Bild', 0, 10, kernel_size)
cv2.createTrackbar('preprocessing_algorithm', 'vorverarbeitetes Bild', 0, 4, preprocessing_algorithm)
cv2.createTrackbar('kernel_size', 'vorverarbeitetes Bild', 0, 3, kernel_size)



# TODO: sobel image:
# slider for threshhold

# TODO: canny image:
# check opencv documentation for threshholds 
# slider for low_threshhold
# slider for high_threshhold
# equalize threshholds

# TODO: XOR image:
# output for sum of pixels (error value)

# TODO: try to minimize error value with sliders


# start program
cv2.waitKey(0)
cv2.destroyAllWindows()