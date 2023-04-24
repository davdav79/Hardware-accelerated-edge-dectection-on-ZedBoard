#TODO: remove the parts outside of the function when done with testing
import cv2
import numpy as np


# define sobel edge detection function
def sobel_edge_detection(img, sigma=1, kernel_size=3, low_threshold=100, high_threshold=200, blur=True):
    # apply gaussian blur
    if blur:
        img_blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    
    # Apply the Sobel operator in the x-direction
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)

    # Apply the Sobel operator in the y-direction
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the magnitude of the gradients
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Threshold the gradient magnitude image
    sobel = np.zeros_like(magnitude)
    sobel[(magnitude >= low_threshold) & (magnitude <= high_threshold)] = 255

    # convert sobel edge detection to binary image
    sobel[sobel > 127] = 255
    sobel[sobel <= 127] = 0
    return sobel

# Load an image
img = cv2.imread("sample_images/Picture_Crossing_noise_10_pixelCnt_65_featureCnt_9.bmp", cv2.IMREAD_GRAYSCALE)

sobel = sobel_edge_detection(img)
# Display the results
cv2.imshow('Original Image', img)
cv2.imshow('Magnitude', sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()