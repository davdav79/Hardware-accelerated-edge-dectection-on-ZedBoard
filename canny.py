import cv2

# define canny edge detection function
def canny_edge_detection(img, sigma=1, kernel_size=3, low_threshold=100, high_threshold=200, blur=True):

    # convert image to 8 bit image
    img = cv2.convertScaleAbs(img)

    # apply gaussian blur
    if blur:
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    # calculate canny edge detection
    canny = cv2.Canny(img, low_threshold, high_threshold)

    # convert canny edge detection to binary image
    canny[canny > 0] = 1

    # return canny edge detection
    return canny