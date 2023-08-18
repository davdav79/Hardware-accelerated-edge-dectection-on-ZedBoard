# 4 interactive windows 1st: original, 2nd preprocessed, 3rd sobel, 4th canny, 5th XOR
import cv2
import numpy as np
import preprocessing as pp
import sobel
import canny


# original image:
# change images with slider (more or less noise)

# window for original image in top left corner
cv2.namedWindow('Originalbild')
cv2.moveWindow('Originalbild', 0, 0)

# function for changing image with noise
def image_change(x):
    # load image with noise
    global img
    img = cv2.imread(f"sample_images/Picture_Crossing_noise_{x*10}_pixelCnt_128_featureCnt_5.bmp")
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # show image with noise
    cv2.imshow('Originalbild', img)

    # call sigma_handler to update preprocessed image
    global sigma
    sigma_handler(sigma)

# add slider for changing image with noise
cv2.createTrackbar('Noise', 'Originalbild', 0, 10, image_change)

# load default image
img = cv2.imread("sample_images/Picture_Crossing_noise_0_pixelCnt_128_featureCnt_5.bmp")
# convert to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# show image with noise
cv2.imshow('Originalbild', img)


# preprocessed image:
# slider for sigma
# slider for kernel_size_blur
# slider for different preprocessing methods (n is dependant on kernel size)
# slider for kernel size

# window for preprocessed image next to Originalbild
cv2.namedWindow('vorverarbeitetes Bild')
cv2.moveWindow('vorverarbeitetes Bild', 100, 0)

# define parameters
sigma = 0
kernel_size_blur = 3
preprocessing_algorithm = 0
kernel_size = 3
preprocessed_img = img

# show image with noise
cv2.imshow('vorverarbeitetes Bild', preprocessed_img)

# functions for preprocessing
def sigma_handler(x):
    # define global variables
    global sigma
    global kernel_size_blur
    global preprocessing_algorithm
    global kernel_size
    global img
    global preprocessed_img

    # set sigma
    sigma = x
    # apply preprocessing
    preprocessed_img = pp.preprocessing(img, sigma, kernel_size_blur, preprocessing_algorithm, kernel_size)
    # show image
    cv2.imshow('vorverarbeitetes Bild', preprocessed_img)

    # update sobel and canny
    global threshold_sobel
    threshold_handler_sobel(threshold_sobel)


def kernel_size_blur_handler(x):
    # define global variables
    global sigma
    global kernel_size_blur
    global preprocessing_algorithm
    global kernel_size
    global img
    global preprocessed_img
    
    # set kernel size
    kernel_size_blur = 2 * x + 3
    # apply preprocessing
    preprocessed_img = pp.preprocessing(img, sigma, kernel_size_blur, preprocessing_algorithm, kernel_size)
    # show image
    cv2.imshow('vorverarbeitetes Bild', preprocessed_img)

    # update sobel and canny
    global threshold_sobel
    threshold_handler_sobel(threshold_sobel)

def preprocessing_algorithm_handler(x):
    # define global variables
    global sigma
    global kernel_size_blur
    global preprocessing_algorithm
    global kernel_size
    global img
    global preprocessed_img
    
    # set kernel size
    preprocessing_algorithm = x
    # apply preprocessing
    preprocessed_img = pp.preprocessing(img, sigma, kernel_size_blur, preprocessing_algorithm, kernel_size)
    # show image
    cv2.imshow('vorverarbeitetes Bild', preprocessed_img)

    if preprocessing_algorithm == 0:
        print("Preprocessing Algorithm: no frame")
    elif preprocessing_algorithm == 1:
        print("Preprocessing Algorithm: absolute frame")
    elif preprocessing_algorithm == 2:
        print("Preprocessing Algorithm: random frame")
    elif preprocessing_algorithm == 3:
        print("Preprocessing Algorithm: mirrored frame")
    elif preprocessing_algorithm == 4:
        print("Preprocessing Algorithm: mirrored oppposite frame")
    elif preprocessing_algorithm == 5:
        print("Preprocessing Algorithm: extrapolated frame")

    # update ideal images
    global ideal_image_sobel_original
    global ideal_image_canny_original
    global ideal_image_sobel
    global ideal_image_canny
    if preprocessing_algorithm != 0:
        ideal_image_sobel = pp.frame_mirror(ideal_image_sobel_original, thickness=int((kernel_size-1)/2)).astype(np.uint8)
        ideal_image_canny = pp.frame_mirror(ideal_image_canny_original, thickness=int((kernel_size-1)/2)).astype(np.uint8)
    else:
        ideal_image_sobel = ideal_image_sobel_original.copy()
        ideal_image_canny = ideal_image_canny_original.copy()

    # update sobel and canny
    global threshold_sobel
    threshold_handler_sobel(threshold_sobel)

def kernel_size_handler(x):
    # define global variables
    global sigma
    global kernel_size_blur
    global preprocessing_algorithm
    global kernel_size
    global img
    global preprocessed_img
    
    # set kernel size
    kernel_size = 2 * x + 3
    # apply preprocessing
    preprocessed_img = pp.preprocessing(img, sigma, kernel_size_blur, preprocessing_algorithm, kernel_size)
    # show image
    cv2.imshow('vorverarbeitetes Bild', preprocessed_img)

    # update ideal images
    global ideal_image_sobel_original
    global ideal_image_canny_original
    global ideal_image_sobel
    global ideal_image_canny
    if preprocessing_algorithm != 0:
        ideal_image_sobel = pp.frame_mirror(ideal_image_sobel_original, thickness=int((kernel_size-1)/2)).astype(np.uint8)
        ideal_image_canny = pp.frame_mirror(ideal_image_canny_original, thickness=int((kernel_size-1)/2)).astype(np.uint8)
    else:
        ideal_image_sobel = ideal_image_sobel_original.copy()
        ideal_image_canny = ideal_image_canny_original.copy()

    # update sobel and canny
    global threshold_sobel
    threshold_handler_sobel(threshold_sobel)

# add sliders for preprocessing
cv2.createTrackbar('sigma', 'vorverarbeitetes Bild', 0, 10, sigma_handler)
cv2.createTrackbar('kernel_size_blur', 'vorverarbeitetes Bild', 0, 10, kernel_size_blur_handler)
cv2.createTrackbar('preprocessing_algorithm', 'vorverarbeitetes Bild', 0, 5, preprocessing_algorithm_handler)



# sobel image:
# slider for threshold

# window for sobel image
cv2.namedWindow('sobel Bild')
cv2.moveWindow('sobel Bild', 200, 0)

# define threshold
threshold_sobel = 0
sobel_img = sobel.sobel_edge_detection(img=preprocessed_img, kernel_size=kernel_size, high_threshold=threshold_sobel, blur=False)

cv2.imshow('sobel Bild', sobel_img*255)

# function for changing threshold
def threshold_handler_sobel(x):
    # define global variables
    global threshold_sobel
    global preprocessed_img
    global kernel_size
    global sobel_img

    # set threshold
    threshold_sobel = x * 10
    # apply sobel
    sobel_img = sobel.sobel_edge_detection(img=preprocessed_img, kernel_size=kernel_size, high_threshold=threshold_sobel, blur=False)
    # show image
    cv2.imshow('sobel Bild', sobel_img*255)

    # update canny
    global threshold_canny
    threshold_handler_canny(cv2.getTrackbarPos('threshold', 'canny Bild'))


# add slider for threshold
cv2.createTrackbar('threshold', 'sobel Bild', 0, 100, threshold_handler_sobel)

# canny image:
# check opencv documentation for thresholds: 
# everything below low_threshold is not an edge
# everything above high_threshold is an edge
# everything between low_threshold and high_threshold is an 
# edge if it is connected to an edge above high_threshold

# slider for low_threshold
# slider for high_threshold
# equalize thresholds

# window for canny image
cv2.namedWindow('canny Bild')
cv2.moveWindow('canny Bild', 300, 0)

# define thresholds
threshold_canny = 0
canny_img = canny.canny_edge_detection(img=preprocessed_img, kernel_size=kernel_size, low_threshold=threshold_canny, high_threshold=threshold_canny, blur=False)

cv2.imshow('canny Bild', canny_img*255)

# function for changing thresholds
def threshold_handler_canny(x):
    # define global variables
    global threshold_canny
    global preprocessed_img
    global kernel_size
    global canny_img

    # set thresholds
    threshold_canny = x * 10
    # apply canny
    canny_img = canny.canny_edge_detection(img=preprocessed_img, kernel_size=kernel_size, low_threshold=threshold_canny, high_threshold=threshold_canny, blur=False)
    # show image
    cv2.imshow('canny Bild', canny_img*255)

    # update XOR image
    XOR_img_handler()

# add slider for thresholds
cv2.createTrackbar('threshold', 'canny Bild', 0, 100, threshold_handler_canny)


# XOR image for difference Sobel - Canny:
# output for sum of pixels (error value)

# window for XOR image
cv2.namedWindow('XOR Bild')
cv2.moveWindow('XOR Bild', 400, 0)

XOR_img = np.bitwise_xor(sobel_img, canny_img)

# calculate XOR image with numpy
def XOR_img_handler():
    global XOR_img
    global sobel_img
    global canny_img
    XOR_img = np.bitwise_xor(sobel_img, canny_img)
    cv2.imshow('XOR Bild', XOR_img*255)

    # print error value
    error_value = np.sum(XOR_img)
    print("XOR Difference Value: " + str(error_value))

    # update XOR error images
    XOR_sobel_err_handler()
    XOR_canny_err_handler()


# XOR image for error Sobel - ideal:
# output for sum of pixels (error value)

# window for XOR image
cv2.namedWindow('XOR Sobel Error Bild')
cv2.moveWindow('XOR Sobel Error Bild', 500, 0)  

ideal_image_sobel = cv2.imread("sample_images/ideal_edges_sobel.bmp")
ideal_image_sobel = cv2.cvtColor(ideal_image_sobel, cv2.COLOR_BGR2GRAY)
_, ideal_image_sobel_original = cv2.threshold(ideal_image_sobel, 0, 1, cv2.THRESH_BINARY)

# make copy of ideal image
ideal_image_sobel = ideal_image_sobel_original.copy()

# calculate XOR image with numpy
XOR_err_sobel = np.bitwise_xor(sobel_img, ideal_image_sobel)

cv2.imshow('XOR Sobel Error Bild', XOR_err_sobel*255)

np.set_printoptions(threshold=np.inf)
# calculate XOR image with numpy
def XOR_sobel_err_handler():
    global XOR_err_sobel
    global sobel_img
    global ideal_image_sobel

    XOR_err_sobel = np.bitwise_xor(sobel_img, ideal_image_sobel)
    cv2.imshow('XOR Sobel Error Bild', XOR_err_sobel*255)

    error_value = np.sum(XOR_err_sobel)

    print("XOR Sobel Error Value: " + str(error_value))

    # calculate TP
    TP = np.sum(np.bitwise_and(sobel_img, ideal_image_sobel))
    # calculate FP
    FP = np.sum(np.bitwise_and(sobel_img, np.bitwise_not(ideal_image_sobel)))
    # calculate FN
    FN = np.sum(np.bitwise_and(np.bitwise_not(sobel_img), ideal_image_sobel))

    # print precision
    print("Sobel Precision: " + str(TP/(TP + FP)))
    # print recall
    print("Sobel Recall: " + str(TP/(TP + FN)))
    # print F1
    print("Sobel F1: " + str(2/(1/(TP/(TP + FP)) + 1/(TP/(TP + FN)))))


# XOR image for error Canny - ideal:
# output for sum of pixels (error value)

# window for XOR image
cv2.namedWindow('XOR Canny Error Bild')
cv2.moveWindow('XOR Canny Error Bild', 600, 0)  

ideal_image_canny = cv2.imread("sample_images/ideal_edges_canny.bmp")
ideal_image_canny = cv2.cvtColor(ideal_image_canny, cv2.COLOR_BGR2GRAY)
_, ideal_image_canny_original = cv2.threshold(ideal_image_canny, 0, 1, cv2.THRESH_BINARY)

# make copy of ideal image
ideal_image_canny = ideal_image_canny_original.copy()

# calculate XOR image with numpy
XOR_err_canny = np.bitwise_xor(canny_img, ideal_image_canny)

cv2.imshow('XOR Canny Error Bild', XOR_err_canny*255)

# calculate XOR image with numpy
def XOR_canny_err_handler():
    global XOR_err_canny
    global canny_img
    global ideal_image_canny
    XOR_err_canny = np.bitwise_xor(canny_img, ideal_image_canny)
    cv2.imshow('XOR Canny Error Bild', XOR_err_canny*255)

    # print error value
    error_value = np.sum(XOR_err_canny)
    print("XOR Canny Error Value: " + str(error_value))
    
    # calculate TP
    TP = np.sum(np.bitwise_and(canny_img, ideal_image_canny))
    # calculate FP
    FP = np.sum(np.bitwise_and(canny_img, np.bitwise_not(ideal_image_canny)))
    # calculate FN
    FN = np.sum(np.bitwise_and(np.bitwise_not(canny_img), ideal_image_canny))

    # print precision
    print("Canny Precision: " + str(TP/(TP + FP)))
    # print recall
    print("Canny Recall: " + str(TP/(TP + FN)))
    # print F1
    print("Canny F1: " + str(2/(1/(TP/(TP + FP)) + 1/(TP/(TP + FN)))))


# small routine to minimize error value (about 5min runtime)
def minimize_error_value(x,y):
    global preprocessed_img
    global kernel_size

    min_error_value = 100000
    min_sobel = 0
    min_canny = 0

    for i in range(255):
        print(i + " / 255")
        for j in range(255):
            # calculate sobel and canny images
            sobel_test_img = sobel.sobel_edge_detection(img=preprocessed_img, kernel_size=kernel_size, high_threshold=i, blur=False)
            canny_test_img = canny.canny_edge_detection(img=preprocessed_img, kernel_size=kernel_size, low_threshold=j, high_threshold=j, blur=False)
            # calculate XOR image
            XOR_test_img = np.bitwise_xor(sobel_img, canny_img)
            error_value = np.sum(XOR_img)
            if error_value < min_error_value:
                min_error_value = error_value
                print("min_error_value: " + str(min_error_value))
                print("min_sobel: " + str(i))
                print("min_canny: " + str(j))
                print('---')

# buttons in buttonbar (press ctrl+p to show buttonbar)
cv2.createButton('minimize error value', minimize_error_value)

# show image
cv2.imshow('XOR Bild', XOR_img*255)

# start program
cv2.waitKey(0)
cv2.destroyAllWindows()