# import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# define function to apply canny edge detection
def canny_edge_detection(img, sigma=1, kernel_size=5, low_threshold=0, high_threshold=0):
    # convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply gaussian blur
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    # calculate sobel edge detection
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # calculate sobel edge detection
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)

    # max value of sobel is about 10000

    # calculate orientation of the edge
    sobel_orientation = np.arctan2(sobel_y, sobel_x)

    # calculate non-maximum suppression
    # edges are suppressed if they are not local maximum
    # this results in thin edges
    output_non_maximum_suppression = np.zeros(sobel.shape)

    for i in range(1, sobel.shape[0] - 1):
        for j in range(1, sobel.shape[1] - 1):
            # get orientation
            orientation = sobel_orientation[i, j]

            # get gradient
            gradient = sobel[i, j]

            # get neighbour pixels
            if (orientation < 0):
                orientation += np.pi

            if (0 <= orientation < np.pi / 4):
                neighbour_1 = sobel[i, j + 1]
                neighbour_2 = sobel[i, j - 1]
            elif (np.pi / 4 <= orientation < np.pi / 2):
                neighbour_1 = sobel[i + 1, j - 1]
                neighbour_2 = sobel[i - 1, j + 1]
            elif (np.pi / 2 <= orientation < 3 * np.pi / 4):
                neighbour_1 = sobel[i + 1, j]
                neighbour_2 = sobel[i - 1, j]
            else:
                neighbour_1 = sobel[i - 1, j - 1]
                neighbour_2 = sobel[i + 1, j + 1]

            # check if gradient is local maximum
            if (gradient >= neighbour_1 and gradient >= neighbour_2):
                output_non_maximum_suppression[i, j] = gradient
            else:
                output_non_maximum_suppression[i, j] = 0

    # calculate double threshold
    # pixels above high_threshold are edges
    # pixels below low_threshold are not edges
    output_double_threshold = np.zeros(output_non_maximum_suppression.shape)

    strong_i, strong_j = np.where(output_non_maximum_suppression >= high_threshold)
    zeros_i, zeros_j = np.where(output_non_maximum_suppression < low_threshold)

    weak_i, weak_j = np.where((output_non_maximum_suppression <= high_threshold) & (output_non_maximum_suppression >= low_threshold))

    output_double_threshold[strong_i, strong_j] = 255
    output_double_threshold[weak_i, weak_j] = 25
    
    # calculate hysteresis
    # pixels between the thresholds are edges if they are connected to a strong pixel
    output_hysteresis = np.zeros(output_double_threshold.shape)
    output_hysteresis[strong_i, strong_j] = 255

    # get weak pixel indices
    weak_i, weak_j = np.where(output_double_threshold == 25)
    
    # loop through weak pixels
    for i, j in zip(weak_i, weak_j):
        # get 3x3 neighbourhood
        neighbourhood = output_double_threshold[i-1:i+2, j-1:j+2]

        # check if any of the neighbours is a strong pixel
        if (255 in neighbourhood):
            output_hysteresis[i, j] = 255

    return output_hysteresis

# read image
#img = cv2.imread('sample_images/Lena.bmp')
img = cv2.imread('sample_images/Picture_Crossing_noise_0_pixelCnt_128_featureCnt_5.bmp')


# make opencv window with sliders for low_threshold, high_threshold and sigma
def nothing(x):
    pass

cv2.namedWindow('Canny Edge Detection')

# everything below low_threshold is not an edge
cv2.createTrackbar('low_threshold', 'Canny Edge Detection', 0, 10000, nothing)
# everything above high_threshold is an edge
cv2.createTrackbar('high_threshold', 'Canny Edge Detection', 0, 10000, nothing)
# everything in between is an edge if it is connected to a high_threshold pixel

# sigma for gaussian blur (basically strength of blur)
cv2.createTrackbar('sigma', 'Canny Edge Detection', 0, 10, nothing)

# add slider to load different images
def load_image(x):
    global img
    if x != 11:
        img = cv2.imread(f"sample_images/Picture_Crossing_noise_{x*10}_pixelCnt_128_featureCnt_5.bmp")
    else:
        img = cv2.imread("sample_images/Lena.bmp")

cv2.createTrackbar('load_image', 'Canny Edge Detection', 0, 11, load_image)


while(1):
    # get current positions of four trackbars
    low_threshold = cv2.getTrackbarPos('low_threshold', 'Canny Edge Detection')
    high_threshold = cv2.getTrackbarPos('high_threshold', 'Canny Edge Detection')
    sigma = cv2.getTrackbarPos('sigma', 'Canny Edge Detection') + 1

    # apply canny edge detection
    output = canny_edge_detection(img, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)

    # show original image
    cv2.imshow('Original Image', img)

    # apply gaussian blur
    blurred = cv2.GaussianBlur(img, (5, 5), sigma)
    # show blurred image
    cv2.imshow('Blurred Image', blurred)

    # show image
    cv2.imshow('Canny Edge Detection', output)

    # wait for key
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
