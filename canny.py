# TODO: implement canny edge detection and convert the result to a binary image

import cv2

# define canny edge detection function
def canny_edge_detection(img, sigma=1, kernel_size=3, low_threshold=100, high_threshold=200, blur=True):
    # convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply gaussian blur
    if blur:
        img_blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    # calculate canny edge detection
    canny = cv2.Canny(img_blur, low_threshold, high_threshold)

    # convert canny edge detection to binary image
    canny[canny > 127] = 255
    canny[canny <= 127] = 0

    # return canny edge detection
    return canny