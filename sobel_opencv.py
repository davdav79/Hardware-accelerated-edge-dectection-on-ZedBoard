import cv2
import numpy as np

# interactive image display for sobel edge detection
def sobel_edge_detection(img, sigma=1, kernel_size_blur=3, kernel_size=3, threshold=100, blur=True):
        
        # apply gaussian blur
        if blur:
            img = cv2.GaussianBlur(img, (kernel_size_blur, kernel_size_blur), sigma)
        
        # Apply the Sobel operator in the x-direction
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)
    
        # Apply the Sobel operator in the y-direction
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)
    
        # Compute the magnitude of the gradients
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Threshold the gradient magnitude image
        sobel = np.zeros_like(magnitude, dtype=np.uint8)
        
        # enter values above high_threshold
        sobel[magnitude > threshold] = 1
    
        # return sobel edge detection
        return sobel

# define sliders for interactive image display
cv2.namedWindow("Sobel Edge Detection")

# define sliders for interactive image display
cv2.createTrackbar("sigma", "Sobel Edge Detection", 0, 10, lambda x: x)
cv2.createTrackbar("kernel_size_blur", "Sobel Edge Detection", 0, 10, lambda x: x)
cv2.createTrackbar("kernel_size", "Sobel Edge Detection", 0, 10, lambda x: x)
cv2.createTrackbar("threshold", "Sobel Edge Detection", 1, 1000, lambda x: x)


# import sample image
img = cv2.imread("sample_images/Picture_Crossing_noise_10_pixelCnt_65_featureCnt_9.bmp")
#img = cv2.imread("sample_images/Lena.bmp")

# convert image to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply sobel edge detection
sobel = sobel_edge_detection(img, blur=False)

while True:
    # get current positions of all trackbars
    sigma = cv2.getTrackbarPos("sigma", "Sobel Edge Detection")
    kernel_size_blur = cv2.getTrackbarPos("kernel_size_blur", "Sobel Edge Detection")
    kernel_size = cv2.getTrackbarPos("kernel_size", "Sobel Edge Detection")
    threshold = cv2.getTrackbarPos("threshold", "Sobel Edge Detection")

    # add offset to sigma
    sigma = sigma + 1

    # make kernel_size_blur and kernel_size odd
    kernel_size_blur = kernel_size_blur*2+1
    kernel_size = kernel_size*2+1

    # adjust threshold to kernel_size
    threshold = threshold*kernel_size**2


    
    # apply sobel edge detection
    sobel = sobel_edge_detection(img, sigma=sigma, kernel_size_blur=kernel_size_blur, kernel_size=kernel_size, threshold=threshold, blur=True)
    
    # display images
    cv2.imshow("Sobel Edge Detection", sobel*255)
    
    # press "q" to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break