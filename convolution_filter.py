import numpy as np
import matplotlib.pyplot as plt

# load bmp image
img = plt.imread("sample_images/Lena.bmp")
#img = plt.imread("sample_images/Picture_Example3_noise_10_pixelCnt_64_featureCnt_7.png")

# convert to grayscale
img = img.mean(axis=2)

# convert to numpy array
img = np.array(img)


# define filters
identity_filter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

edge_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

sobel_x_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

gaussian_filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

# define convolution function
def convolution(img, conv_filter):
    # get image size
    img_row, img_col = img.shape

    # get filter size
    filter_row, filter_col = conv_filter.shape

    # create output image with 1 extra line and column for the frame
    output = np.zeros((img_row, img_col))

    # convolution
    for i in range(img_row - filter_row+1):
        for j in range(img_col - filter_col+1):
            output[i+1, j+1] = np.sum(img[i:i + filter_row, j:j + filter_col] * conv_filter)

    # return output
    return output

# apply convolution
output = convolution(img, edge_filter)

output_gaussian = convolution(img, gaussian_filter)

output_sobel_x = convolution(output_gaussian, sobel_x_filter)
output_sobel_y = convolution(output_gaussian, sobel_y_filter)

# calculate sobel edge detection
output_sobel = np.sqrt(output_sobel_x**2 + output_sobel_y**2)

# calculate orientation of the edge
output_sobel_orientation = np.arctan2(output_sobel_y, output_sobel_x)


# show image and output and difference image
plt.subplot(2, 4, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")

plt.subplot(2, 4, 2)
plt.imshow(output, cmap="gray")
plt.title("Output Image primitive edge detection")

plt.subplot(2, 4, 3)
plt.imshow(output_sobel, cmap="gray")
plt.title("Output Image sobel edge detection")

plt.subplot(2, 4, 4)
plt.imshow(output_sobel_orientation, cmap="rainbow")
plt.title("Output Image sobel edge detection orientation")

plt.subplot(2, 4, 5)
plt.imshow(output_gaussian, cmap="gray")
plt.title("Output Image gaussian filter")

plt.subplot(2, 4, 6)
plt.imshow(output_sobel_x, cmap="gray")
plt.title("Output Image sobel x filter")

plt.subplot(2, 4, 7)
plt.imshow(output_sobel_y, cmap="gray")
plt.title("Output Image sobel y filter")

plt.show()