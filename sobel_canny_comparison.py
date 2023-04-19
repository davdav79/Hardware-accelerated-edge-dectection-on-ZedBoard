

# TODO: compare the results of the Sobel and Canny edge detection algorithms by XORing the binary images
# TODO: plots the results of the Sobel and Canny edge detection algorithms in a single figure
# TODO: try different preprocessing steps and compare the results

import matplotlib.pyplot as plt
import cv2

import preprocessing as pp

# import sample image
#img = cv2.imread("sample_images/Picture_Crossing_noise_10_pixelCnt_65_featureCnt_9.bmp")
img = cv2.imread("sample_images/Lena.bmp")

# convert image to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#img = pp.frame_absolute(img, 127, 10)
#img = pp.frame_random(img, 10)

#img = pp.frame_mirror(img, 10)

#img = pp.frame_mirror_opposite(img, 10)

img = pp.frame_extrapolate(img, 10)

# show image
plt.imshow(img, cmap='gray')
plt.show()