# import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# define sobel edge detection function
def sobel_edge_detection(img, sigma=1, kernel_size=3):
    # convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply gaussian blur
    img_blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    # calculate sobel edge detection
    sobel_x = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # calculate sobel edge detection
    sobelxy = np.sqrt(sobel_x**2 + sobel_y**2)

    # calculate orientation of the edge
    sobel_orientation = np.arctan2(sobel_y, sobel_x)

    # return sobel edge detection
    return sobelxy, sobel_orientation


cv2.namedWindow('Sobel Edge Detection')

# create trackbar for sigma
cv2.createTrackbar('sigma', 'Sobel Edge Detection', 0, 10, lambda x: None)


# add slider to load different images
def load_image(x):
    global img
    if x != 11:
        img = cv2.imread(f"sample_images/Picture_Crossing_noise_{x*10}_pixelCnt_128_featureCnt_5.bmp")
    else:
        img = cv2.imread("sample_images/Lena.bmp")

cv2.createTrackbar('load_image', 'Sobel Edge Detection', 0, 11, load_image)


while(1):
    # get current positions of four trackbars
    load_image(cv2.getTrackbarPos('load_image', 'Sobel Edge Detection'))
    sigma = cv2.getTrackbarPos('sigma', 'Sobel Edge Detection')

    # apply sobel edge detection
    sobel, sobel_orientation = sobel_edge_detection(img, sigma=sigma)

    # show original image
    cv2.imshow('Original Image', img)

    # apply gaussian blur
    blurred = cv2.GaussianBlur(img, (5, 5), sigma)
    # show blurred image
    cv2.imshow('Blurred Image', blurred)

    # show image
    sobel = (sobel - np.min(sobel)) / (np.max(sobel) - np.min(sobel)) * 255
    sobel = sobel.astype(np.uint8)
    cv2.imshow('Sobel Edge Detection', sobel)


    # show oriantation image in color
    sobel_orientation = (sobel_orientation + np.pi) / (2 * np.pi) * 255
    sobel_orientation = sobel_orientation.astype(np.uint8)
    sobel_orientation = cv2.applyColorMap(sobel_orientation, cv2.COLORMAP_JET)
    cv2.imshow('Sobel Orientation', sobel_orientation)

    # wait for key
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()