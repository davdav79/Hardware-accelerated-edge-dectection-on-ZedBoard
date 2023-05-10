
import cv2
import numpy as np

# define function for adding frame consisting of absolute values (0, average, 255)
def frame_absolute(img, fill_value=0, thickness=1):
    # get image shape
    img_shape = img.shape

    # create frame
    frame = np.ones((img_shape[0] + 2 * thickness, img_shape[1] + 2 * thickness)) * fill_value

    # add image to frame
    frame[thickness:-thickness, thickness:-thickness] = img

    # return frame
    return frame

# TODO fix when blur is applied
# define function for adding frame consisting of random values
def frame_random(img, thickness=1):
    # get image shape
    img_shape = img.shape

    # create frame
    frame = np.random.randint(0, 256, (img_shape[0] + 2 * thickness, img_shape[1] + 2 * thickness))

    # add image to frame
    frame[thickness:-thickness, thickness:-thickness] = img

    # return frame
    return frame

# define function for adding frame by extrapolating values by adding gradient of adjacent pixel
def frame_extrapolate(img, thickness=1):
    # get image shape
    img_shape = img.shape

    # create frame
    frame = np.zeros((img_shape[0] + 2 * thickness, img_shape[1] + 2 * thickness))

    # add image to frame
    frame[thickness:-thickness, thickness:-thickness] = img

    # calculate gradient between first and second line
    gradient_top = frame[thickness+1, thickness:-thickness] - frame[thickness, thickness:-thickness]

    # calculate gradient between first and second column
    gradient_left = frame[thickness:-thickness, thickness + 1] - frame[thickness:-thickness, thickness]

    # calculate gradient between last and second last line
    gradient_bottom = frame[-thickness - 2, thickness:-thickness] - frame[-thickness - 1, thickness:-thickness]

    # calculate gradient between last and second last column
    gradient_right = frame[thickness:-thickness, -thickness - 2] - frame[thickness:-thickness, -thickness - 1]

    # calculate gradient for top left corner
    gradient_top_left = frame[thickness + 1, thickness + 1] - frame[thickness, thickness]

    # calculate gradient for top right corner
    gradient_top_right = frame[thickness + 1, -thickness - 2] - frame[thickness, -thickness - 1]

    # calculate gradient for bottom left corner
    gradient_bottom_left = frame[-thickness - 2, thickness + 1] - frame[-thickness - 1, thickness]

    # calculate gradient for bottom right corner
    gradient_bottom_right = frame[-thickness - 2, -thickness - 2] - frame[-thickness - 1, -thickness - 1]


    # extrapolate top frame by adding gradient continuously
    for i in range(thickness):
        frame[thickness-i-1, thickness:-thickness] = frame[thickness-i, thickness:-thickness] - gradient_top

    # extrapolate left frame
    for i in range(thickness):
        frame[thickness:-thickness, thickness-i-1] = frame[thickness:-thickness, thickness-i] - gradient_left

    # extrapolate bottom frame
    for i in range(thickness):
        frame[-thickness + i, thickness:-thickness] = frame[-thickness + i - 1, thickness:-thickness] - gradient_bottom

    # extrapolate right frame
    for i in range(thickness):
        frame[thickness:-thickness, -thickness + i ] = frame[thickness:-thickness, -thickness + i - 1] - gradient_right

    # extrapolate corners by extrapolating diagonally
    for i in range(thickness):
        frame[thickness-i-1, thickness-i-1] = frame[thickness-i, thickness-i] - gradient_top_left
        frame[thickness-i-1, -thickness+i] = frame[thickness-i, -thickness+i-1] - gradient_top_right
        frame[-thickness+i, thickness-i-1] = frame[-thickness+i-1, thickness-i] - gradient_bottom_left
        frame[-thickness+i, -thickness+i] = frame[-thickness+i-1, -thickness+i-1] - gradient_bottom_right

    # cut negative values
    frame[frame < 0] = 0

    # cut values above 255
    frame[frame > 255] = 255

    # calculate for each extrapolated corner diagonal the gradient between the corner and the last line/column
    gradient_top_left_left = []
    gradient_top_left_right = []
    gradient_top_right_left = []
    gradient_top_right_right = []
    gradient_bottom_left_left = []
    gradient_bottom_left_right = []
    gradient_bottom_right_left = []
    gradient_bottom_right_right = []

    for i in range(thickness-1):
        gradient_top_right_left.append(frame[i,-i-1]-frame[i,-thickness-1])
        gradient_top_right_right.append(frame[i,-i-1]-frame[thickness,-i-1])
        
        gradient_top_left_left.append(frame[i,i]-frame[thickness,i])
        gradient_top_left_right.append(frame[i,i]-frame[i,thickness])

        gradient_bottom_right_left.append(frame[-i-1,-i-1]-frame[-i-1,-thickness-1])
        gradient_bottom_right_right.append(frame[-i-1,-i-1]-frame[-thickness-1,-i-1])

        gradient_bottom_left_left.append(frame[-i-1,i]-frame[-thickness-1,i])
        gradient_bottom_left_right.append(frame[-i-1,i]-frame[-i-1,thickness])
    

    # fill missing corner values by interpolating between diagonal and values of last line/column
    for i in range (thickness-1):
        # generate interpolation values
        
        interpolation_top_right_left = np.linspace(0, 1, thickness-i+1) * gradient_top_right_left[i] + frame[i, -thickness-1]
        interpolation_top_right_right = np.linspace(0, 1, thickness-i+1) * gradient_top_right_right[i] + frame[thickness, -i-1]
        
        interpolation_top_left_left = np.linspace(0, 1, thickness-i+1) * gradient_top_left_left[i] + frame[thickness, i]
        interpolation_top_left_right = np.linspace(0, 1, thickness-i+1) * gradient_top_left_right[i] + frame[i, thickness]

        interpolation_bottom_right_left = np.linspace(0, 1, thickness-i+1) * gradient_bottom_right_left[i] + frame[-i-1,-thickness-1]
        interpolation_bottom_right_right = np.linspace(0, 1, thickness-i+1) * gradient_bottom_right_right[i] + frame[-thickness-1,-i-1]

        interpolation_bottom_left_left = np.linspace(0, 1, thickness-i+1) * gradient_bottom_left_left[i] + frame[-thickness-1,i]
        interpolation_bottom_left_right = np.linspace(0, 1, thickness-i+1) * gradient_bottom_left_right[i] + frame[-i-1,thickness]

        # fill missing values
        frame[i, -thickness:-i-1] = interpolation_top_right_left[1:-1]
        frame[i+1:thickness, -i-1] = interpolation_top_right_right[1:-1]
        
        frame[i, i+1:thickness] = interpolation_top_left_left[1:-1]
        frame[i+1:thickness, i] = interpolation_top_left_right[1:-1]

        frame[-i-1, -thickness:-i-1] = interpolation_bottom_right_left[1:-1]
        frame[-thickness:-i-1, -i-1] = interpolation_bottom_right_right[1:-1]

        frame[-i-1, i+1:thickness] = interpolation_bottom_left_left[1:-1]
        frame[-thickness:-i-1, i] = interpolation_bottom_left_right[1:-1]
        



    # cut negative values
    frame[frame < 0] = 0

    # cut values above 255
    frame[frame > 255] = 255


    # return frame
    return frame

# TODO: check if edges are mirrored and not copied
# define function for adding frame by mirroring last line and averaging corners
def frame_mirror(img, thickness=1):
    # get image shape
    img_shape = img.shape

    # create frame
    frame = np.zeros((img_shape[0] + 2 * thickness, img_shape[1] + 2 * thickness))

    # add image to frame
    frame[thickness:-thickness, thickness:-thickness] = img

    # mirror last line
    frame[:thickness, thickness:-thickness] = frame[thickness, thickness:-thickness]
    frame[-thickness:, thickness:-thickness] = frame[-thickness - 1, thickness:-thickness]

    # mirror first column
    frame[thickness:-thickness, :thickness] = frame[thickness:-thickness, thickness].reshape(-1, 1)
    frame[thickness:-thickness, -thickness:] = frame[thickness:-thickness, -thickness - 1].reshape(-1, 1)

    # average corners
    frame[:thickness, :thickness] = (frame[thickness, thickness] + frame[thickness, -thickness - 1]) / 2
    frame[:thickness, -thickness:] = (frame[thickness, thickness] + frame[thickness, -thickness - 1]) / 2
    frame[-thickness:, :thickness] = (frame[-thickness - 1, thickness] + frame[-thickness - 1, -thickness - 1]) / 2
    frame[-thickness:, -thickness:] = (frame[-thickness - 1, thickness] + frame[-thickness - 1, -thickness - 1]) / 2

    # return frame
    return frame

# define function for adding frame by mirroring opposite line and averaging corners
def frame_mirror_opposite(img, thickness=1):
    # get image shape
    img_shape = img.shape

    # create frame
    frame = np.zeros((img_shape[0] + 2 * thickness, img_shape[1] + 2 * thickness))

    # add image to frame
    frame[thickness:-thickness, thickness:-thickness] = img

    # mirror last line
    frame[:thickness, thickness:-thickness] = frame[-thickness - 1, thickness:-thickness]
    frame[-thickness:, thickness:-thickness] = frame[thickness, thickness:-thickness]

    # mirror first column
    frame[thickness:-thickness, :thickness] = frame[thickness:-thickness, -thickness - 1].reshape(-1, 1)
    frame[thickness:-thickness, -thickness:] = frame[thickness:-thickness, thickness].reshape(-1, 1)

    # average corners
    frame[:thickness, :thickness] = (frame[-thickness - 1, thickness] + frame[-thickness - 1, -thickness - 1]) / 2
    frame[:thickness, -thickness:] = (frame[-thickness - 1, thickness] + frame[-thickness - 1, -thickness - 1]) / 2
    frame[-thickness:, :thickness] = (frame[thickness, thickness] + frame[thickness, -thickness - 1]) / 2
    frame[-thickness:, -thickness:] = (frame[thickness, thickness] + frame[thickness, -thickness - 1]) / 2


    # return frame
    return frame


# function for gaussian blur
def gaussian_blur(img, kernel_size=3, sigma=1,borderType=cv2.BORDER_DEFAULT):
    # apply gaussian blur
    img_blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma,borderType=borderType)

    # return blurred image
    return img_blur


def preprocessing(img, sigma, kernel_size_blur, preprocessing_algorithm, kernel_size,borderType=cv2.BORDER_DEFAULT):
    # apply preprocessing algorithm
    if preprocessing_algorithm == 1:
        img = frame_absolute(img=img, thickness=int((kernel_size-1)/2))     
    elif preprocessing_algorithm == 2:
        img = frame_random(img, thickness=int((kernel_size-1)/2))
    elif preprocessing_algorithm == 3:
        img = frame_mirror(img, thickness=int((kernel_size-1)/2))
    elif preprocessing_algorithm == 4:
        img = frame_mirror_opposite(img, thickness=int((kernel_size-1)/2))
    elif preprocessing_algorithm == 5:
        img = frame_extrapolate(img, thickness=int((kernel_size-1)/2))

    # apply gaussian blur
    if sigma != 0:
        img = gaussian_blur(img, kernel_size_blur, sigma,borderType)

    if preprocessing_algorithm == 0:
        return img

    img = img.astype(np.uint8)

    # return preprocessed image
    return img