import matplotlib.pyplot as plt
import cv2

import preprocessing as pp
import sobel
import canny

# import sample image
img = cv2.imread("sample_images/Picture_Crossing_noise_10_pixelCnt_65_featureCnt_9.bmp")
#img = cv2.imread("sample_images/Lena.bmp")

# convert image to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# apply preprocessing
img = pp.frame_mirror(img, 1)

# apply sobel edge detection
sobel = sobel.sobel_edge_detection(img, blur=False)

# apply canny edge detection
canny = canny.canny_edge_detection(img, blur=False)

# XOR sobel and canny
xor = cv2.bitwise_xor(sobel, canny)


# plot original image, sobel, canny and XOR
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.subplot(2, 2, 2)
plt.imshow(sobel)
plt.title("Sobel")
plt.subplot(2, 2, 3)
plt.imshow(canny)
plt.title("Canny")
plt.subplot(2, 2, 4)
plt.imshow(xor)
plt.title("XOR")
plt.show()
