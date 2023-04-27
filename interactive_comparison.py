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
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # show image with noise
    cv2.imshow('Originalbild', img)
    global sigma
    sigma_handler(sigma)

# add slider for changing image with noise
cv2.createTrackbar('Noise', 'Originalbild', 0, 10, image_change)

# load default image
img = cv2.imread("sample_images/Picture_Crossing_noise_0_pixelCnt_128_featureCnt_5.bmp")
# convert to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# show image with noise
cv2.imshow('Originalbild', img)


# TODO: preprocessed image:
# slider for sigma
# slider for kernel_size_blur
# slider for different preprocessing methods (n is dependant on kernel size)
# slider for kernel size

# window for preprocessed image
cv2.namedWindow('vorverarbeitetes Bild')

# load default image
img = cv2.imread("sample_images/Picture_Crossing_noise_0_pixelCnt_128_featureCnt_5.bmp")
# convert to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# show image with noise
cv2.imshow('vorverarbeitetes Bild', img)

# define parameters
sigma = 1
kernel_size_blur = 3
preprocessing_algorithm = 0
kernel_size = 3

# functions for preprocessing
def sigma_handler(x):
    # define global variables
    global sigma
    global kernel_size_blur
    global preprocessing_algorithm
    global kernel_size
    global img

    # set sigma
    sigma = x
    # apply preprocessing
    preprocessed_img = pp.preprocessing(img, sigma, kernel_size_blur, preprocessing_algorithm, kernel_size)
    # show image
    cv2.imshow('vorverarbeitetes Bild', preprocessed_img)


def kernel_size_blur_handler(x):
    # define global variables
    global sigma
    global kernel_size_blur
    global preprocessing_algorithm
    global kernel_size
    global img
    
    # set kernel size
    kernel_size_blur = 2 * x + 3
    # apply preprocessing
    preprocessed_img = pp.preprocessing(img, sigma, kernel_size_blur, preprocessing_algorithm, kernel_size)
    # show image
    cv2.imshow('vorverarbeitetes Bild', preprocessed_img)

def preprocessing_algorithm_handler(x):
    # define global variables
    global sigma
    global kernel_size_blur
    global preprocessing_algorithm
    global kernel_size
    global img
    
    # set kernel size
    preprocessing_algorithm = x
    # apply preprocessing
    preprocessed_img = pp.preprocessing(img, sigma, kernel_size_blur, preprocessing_algorithm, kernel_size)
    # show image
    cv2.imshow('vorverarbeitetes Bild', preprocessed_img)

def kernel_size_handler(x):
    # define global variables
    global sigma
    global kernel_size_blur
    global preprocessing_algorithm
    global kernel_size
    global img
    
    # set kernel size
    kernel_size = 2 * x + 3
    # apply preprocessing
    preprocessed_img = pp.preprocessing(img, sigma, kernel_size_blur, preprocessing_algorithm, kernel_size)
    # show image
    cv2.imshow('vorverarbeitetes Bild', preprocessed_img)

# add sliders for preprocessing
cv2.createTrackbar('sigma', 'vorverarbeitetes Bild', 0, 10, sigma_handler)
cv2.createTrackbar('kernel_size_blur', 'vorverarbeitetes Bild', 0, 10, kernel_size_blur_handler)
cv2.createTrackbar('preprocessing_algorithm', 'vorverarbeitetes Bild', 0, 5, preprocessing_algorithm_handler)
cv2.createTrackbar('kernel_size', 'vorverarbeitetes Bild', 0, 3, kernel_size_handler)

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