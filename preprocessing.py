
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

# define function for adding frame by extrapolating values by adding gradient of adjacent pixel #TODO: fix this
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


    # extrapolate top frame by continuing gradient
    for i in range(thickness):
        frame[thickness-i-1, thickness:-thickness] = frame[thickness-i, thickness:-thickness] + gradient_top

    # extrapolate left frame
    for i in range(thickness):
        frame[thickness:-thickness, i] = frame[thickness:-thickness, i + 1] + gradient_left

    # extrapolate bottom frame
    for i in range(thickness):
        frame[-i - 1, thickness:-thickness] = frame[-i - 2, thickness:-thickness] + gradient_bottom

    # extrapolate right frame
    for i in range(thickness):
        frame[thickness:-thickness, -i - 1] = frame[thickness:-thickness, -i - 2] + gradient_right


    # return frame
    return frame

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
def gaussian_blur(img, kernel_size=3, sigma=1):
    # apply gaussian blur
    img_blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

    # return blurred image
    return img_blur
