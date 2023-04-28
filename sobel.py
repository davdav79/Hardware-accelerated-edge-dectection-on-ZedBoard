import cv2
import numpy as np

# define sobel edge detection function
def sobel_edge_detection(img, sigma=1, kernel_size=3, low_threshold=100, high_threshold=200, blur=True):
    # apply gaussian blur
    if blur:
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    
    # Apply the Sobel operator in the x-direction
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

    # Apply the Sobel operator in the y-direction
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the magnitude of the gradients
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Threshold the gradient magnitude image
    sobel = np.zeros_like(magnitude, dtype=np.uint8)
    
    # enter values above high_threshold
    sobel[magnitude > high_threshold] = 1

    # return sobel edge detection
    return sobel