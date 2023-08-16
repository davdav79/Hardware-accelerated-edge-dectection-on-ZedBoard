import numpy as np
import matplotlib.pyplot as plt

# load bmp image
raw_img = plt.imread("sample_images/Lena.bmp")
#img = plt.imread("sample_images/Picture_Example3_noise_10_pixelCnt_64_featureCnt_7.png")

# convert to grayscale
raw_img = raw_img.mean(axis=2)

# convert to numpy array
img = np.array(raw_img)


# define filters
identity_filter = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

primitive_edge_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

sobel_x_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y_filter = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

gaussian_filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
median_filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9

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
output_median = convolution(img, median_filter)

output_sobel_x = convolution(output_gaussian, sobel_x_filter)
output_sobel_y = convolution(output_gaussian, sobel_y_filter)

# calculate sobel edge detection
output_sobel = np.sqrt(output_sobel_x**2 + output_sobel_y**2)

# calculate orientation of the edge
output_sobel_orientation = np.arctan2(output_sobel_y, output_sobel_x)


# stretch images to 0-255
# show image and output and difference image
plt.subplot(2, 4, 1)
plt.imshow(raw_img, cmap="gray", vmin=0, vmax=255)
plt.title("Original Image")

plt.subplot(2, 4, 2)
plt.imshow(output, cmap="gray", vmin=0, vmax=255)
plt.title("Output Image primitive edge detection")

plt.subplot(2, 4, 3)
plt.imshow(output_sobel, cmap="gray", vmin=0, vmax=255)
plt.title("Output Image sobel edge detection")

plt.subplot(2, 4, 4)
plt.imshow(output_sobel_orientation, cmap="rainbow")
plt.title("Output Image sobel edge detection orientation")

plt.subplot(2, 4, 5)
plt.imshow(output_gaussian, cmap="gray", vmin=0, vmax=255)
plt.title("Output Image gaussian filter")

plt.subplot(2, 4, 6)
plt.imshow(output_sobel_x, cmap="gray", vmin=0, vmax=255)
plt.title("Output Image sobel x filter")

plt.subplot(2, 4, 7)
plt.imshow(output_sobel_y, cmap="gray", vmin=0, vmax=255)
plt.title("Output Image sobel y filter")

plt.subplot(2, 4, 8)
plt.imshow(output_median, cmap="gray", vmin=0, vmax=255)
plt.title("Output Image median filter")


# save original image, median and gaussian
plt.imsave("Lena_original.png", raw_img, cmap="gray", vmin=0, vmax=255)
plt.imsave("Lena_median.png", output_median, cmap="gray", vmin=0, vmax=255)
plt.imsave("Lena_gaussian.png", output_gaussian, cmap="gray", vmin=0, vmax=255)

output_sobel_gaussian_x = convolution(output_gaussian, sobel_x_filter)
output_sobel_gaussian_y = convolution(output_gaussian, sobel_y_filter)

output_sobel_gaussian = np.sqrt(output_sobel_gaussian_x**2 + output_sobel_gaussian_y**2)

# set values below 100 to 0 and above 200 to 255
output_sobel_binary = np.zeros_like(output_sobel_gaussian, dtype=np.uint8)
output_sobel_binary[output_sobel_gaussian > 100] = 255

# Non max supression
output_sobel_gaussian_orientation = np.arctan2(output_sobel_gaussian_y, output_sobel_gaussian_x)

output_non_max_supression = np.zeros_like(output_sobel_gaussian, dtype=np.uint8)

i = 1
j = 1
while i < output_sobel_gaussian_orientation.shape[0]-1:
    j = 1
    while j < output_sobel_gaussian_orientation.shape[1]-1:
        if output_sobel_gaussian_orientation[i, j] >= -22.5 * np.pi / 180 and output_sobel_gaussian_orientation[i, j] < 22.5 * np.pi / 180 and output_sobel_gaussian[i, j] > output_sobel_gaussian[i, j-1] and output_sobel_gaussian[i, j] > output_sobel_gaussian[i, j+1]:
            output_non_max_supression[i, j] = output_sobel_gaussian[i, j]
        elif output_sobel_gaussian_orientation[i, j] >= 22.5 * np.pi / 180 and output_sobel_gaussian_orientation[i, j] < 67.5 * np.pi / 180 and output_sobel_gaussian[i, j] > output_sobel_gaussian[i-1, j-1] and output_sobel_gaussian[i, j] > output_sobel_gaussian[i+1, j+1]:
            output_non_max_supression[i, j] = output_sobel_gaussian[i, j]
        elif output_sobel_gaussian_orientation[i, j] >= 67.5 * np.pi / 180 and output_sobel_gaussian_orientation[i, j] < 112.5 * np.pi / 180 and output_sobel_gaussian[i, j] > output_sobel_gaussian[i-1, j] and output_sobel_gaussian[i, j] > output_sobel_gaussian[i+1, j]:
            output_non_max_supression[i, j] = output_sobel_gaussian[i, j]
        elif output_sobel_gaussian_orientation[i, j] >= 112.5 * np.pi / 180 and output_sobel_gaussian_orientation[i, j] < 157.5 * np.pi / 180 and output_sobel_gaussian[i, j] > output_sobel_gaussian[i-1, j+1] and output_sobel_gaussian[i, j] > output_sobel_gaussian[i+1, j-1]:
            output_non_max_supression[i, j] = output_sobel_gaussian[i, j]
        else:
            output_non_max_supression[i, j] = 0
        j += 1
    i += 1


output_canny_binary = np.zeros_like(output_sobel_gaussian, dtype=np.uint8)

high_threshold = 100
low_threshold = 30

# apply hysteresis threshold itteratively
i = 1
j = 1
while i < output_sobel_gaussian_orientation.shape[0]-1:
    j = 1
    while j < output_sobel_gaussian_orientation.shape[1]-1:
        if output_non_max_supression[i, j] > high_threshold:
            output_canny_binary[i, j] = 255
        j += 1
    i += 1




# print number of weak and strong pixels
print("number of strong pixels: ", np.sum(output_canny_binary == 255))
# print number of pixel between low_threshold and high_threshold
print("number of weak pixels: ", np.sum(output_non_max_supression < high_threshold) - np.sum(output_non_max_supression > low_threshold))

# copy output_non_max_supression
output_canny_buffer = np.copy(output_non_max_supression)

stop = 0
counter = 0
while stop == 0:
    # iteratre through high threshold pixels
    i = 1
    j = 1
    stop = 1
    print(counter)
    while i < output_sobel_gaussian_orientation.shape[0]-1:
        j = 1
        while j < output_sobel_gaussian_orientation.shape[1]-1:
            if output_canny_binary[i, j] == 255:
                # check if any of the 8 neighbours is above the low threshold
                if output_canny_buffer[i-1, j-1] > low_threshold and output_canny_buffer[i-1, j-1] < high_threshold:
                    output_canny_binary[i-1, j-1] = 255
                    output_canny_buffer[i-1, j-1] = 0
                    stop = 0
                    counter += 1
                elif output_canny_buffer[i-1, j] > low_threshold and output_canny_buffer[i-1, j] < high_threshold:
                    output_canny_binary[i-1, j] = 255
                    output_canny_buffer[i-1, j] = 0
                    stop = 0
                    counter += 1
                elif output_canny_buffer[i-1, j+1] > low_threshold and output_canny_buffer[i-1, j+1] < high_threshold:
                    output_canny_binary[i-1, j+1] = 255
                    output_canny_buffer[i-1, j+1] = 0
                    stop = 0
                    counter += 1
                elif output_canny_buffer[i, j-1] > low_threshold and output_canny_buffer[i, j-1] < high_threshold:
                    output_canny_binary[i, j-1] = 255
                    output_canny_buffer[i, j-1] = 0
                    stop = 0
                    counter += 1
                elif output_canny_buffer[i, j+1] > low_threshold and output_canny_buffer[i, j+1] < high_threshold:
                    output_canny_binary[i, j+1] = 255
                    output_canny_buffer[i, j+1] = 0
                    stop = 0
                    counter += 1
                elif output_canny_buffer[i+1, j-1] > low_threshold and output_canny_buffer[i+1, j-1] < high_threshold:
                    output_canny_binary[i+1, j-1] = 255
                    output_canny_buffer[i+1, j-1] = 0
                    stop = 0
                    counter += 1
                elif output_canny_buffer[i+1, j] > low_threshold and output_canny_buffer[i+1, j] < high_threshold:
                    output_canny_binary[i+1, j] = 255
                    output_canny_buffer[i+1, j] = 0
                    stop = 0
                    counter += 1
                elif output_canny_buffer[i+1, j+1] > low_threshold and output_canny_buffer[i+1, j+1] < high_threshold:
                    output_canny_binary[i+1, j+1] = 255
                    output_canny_buffer[i+1, j+1] = 0
                    stop = 0
                    counter += 1
            j += 1
        i += 1




plt.imsave("Lena_sobel_gaussian_canny_binary.png", output_canny_binary, cmap="gray")

plt.imsave("Lena_sobel_gaussian_orientation.png", output_sobel_gaussian_orientation, cmap="rainbow")


plt.imsave("Lena_sobel_gaussian_non_max_suppression.png", output_non_max_supression, cmap="gray", vmin=0, vmax=255)
plt.imsave("Lena_sobel_gaussian_x.png", output_sobel_gaussian_x, cmap="gray", vmin=0, vmax=255)
plt.imsave("Lena_sobel_gaussian_y.png", output_sobel_gaussian_y, cmap="gray", vmin=0, vmax=255)
plt.imsave("Lena_sobel_gaussian.png", output_sobel_gaussian, cmap="gray", vmin=0, vmax=255)
plt.imsave("Lena_sobel_gaussian_binary.png", output_sobel_binary, cmap="gray", vmin=0, vmax=255)

plt.show()