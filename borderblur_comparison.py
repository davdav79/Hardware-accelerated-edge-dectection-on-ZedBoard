# 4 interactive windows 1st: original, 2nd preprocessed, 3rd sobel, 4th canny, 5th XOR
import cv2
import numpy as np
import preprocessing as pp
import sobel
import canny

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
cv2.namedWindow('vorverarbeitetes Bild no border')
cv2.moveWindow('vorverarbeitetes Bild no border', 100, 0)


# define parameters
sigma = 0
kernel_size_blur = 3
preprocessing_algorithm = 0
kernel_size = 3
preprocessed_img = img
preprocessed_img_no_border = img

# show image with noise
cv2.imshow('vorverarbeitetes Bild', preprocessed_img)
cv2.imshow('vorverarbeitetes Bild no border', preprocessed_img_no_border)

# define threshold
threshold_sobel = 0
sobel_img = sobel.sobel_edge_detection(img=preprocessed_img, kernel_size=kernel_size, high_threshold=threshold_sobel, blur=False)
sobel_img_no_border = sobel.sobel_edge_detection(img=preprocessed_img_no_border, kernel_size=kernel_size, high_threshold=threshold_sobel, blur=False)

# define thresholds
threshold_canny = 0
canny_img = canny.canny_edge_detection(img=preprocessed_img, kernel_size=kernel_size, low_threshold=threshold_canny, high_threshold=threshold_canny, blur=False)
canny_img_no_border = canny.canny_edge_detection(img=preprocessed_img, kernel_size=kernel_size, low_threshold=threshold_canny, high_threshold=threshold_canny, blur=False)


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

sobel_canny = 0
# function for changing threshold
def threshold_handler_sobel(x):
    # define global variables
    global threshold_sobel
    global preprocessed_img
    global preprocessed_img_no_border
    global kernel_size
    global sobel_img
    global sobel_img_no_border

    # set threshold
    threshold_sobel = x*kernel_size**2
    # apply sobel
    sobel_img = sobel.sobel_edge_detection(img=preprocessed_img, kernel_size=kernel_size, high_threshold=threshold_sobel, blur=False)
    sobel_img_no_border = sobel.sobel_edge_detection(img=preprocessed_img_no_border, kernel_size=kernel_size, high_threshold=threshold_sobel, blur=False)
    # show image
    cv2.imshow('sobel Bild', sobel_img*255)
    cv2.imshow('sobel Bild no border', sobel_img*255)
    XOR_img_handler()
    # update canny
    ##global threshold_canny
    ##threshold_handler_canny(cv2.getTrackbarPos('threshold', 'canny Bild'))

# function for changing thresholds
def threshold_handler_canny(x):
    # define global variables
    global threshold_canny
    global preprocessed_img
    global preprocessed_img_no_border
    global kernel_size
    global canny_img
    global canny_img_no_border

    # set thresholds
    threshold_canny = x*kernel_size**2
    # apply canny
    canny_img = canny.canny_edge_detection(img=preprocessed_img, kernel_size=kernel_size, low_threshold=threshold_canny, high_threshold=threshold_canny, blur=False)
    canny_img_no_border = canny.canny_edge_detection(img=preprocessed_img_no_border, kernel_size=kernel_size, low_threshold=threshold_canny, high_threshold=threshold_canny, blur=False)
    # show image
    cv2.imshow('canny Bild', canny_img*255)
    cv2.imshow('canny Bild', canny_img_no_border*255)

    # update XOR image
    XOR_img_handler()


def sobel_canny_draw(x):
    global sobel_canny
    global sobel_img
    global sobel_img_no_border
    global canny_img
    global canny_img_no_border
    sobel_canny = x
    if(x == 0):
        if(cv2.getWindowProperty('canny Bild', cv2.WND_PROP_VISIBLE) == -1):
            print("destroy windosc")
            cv2.destroyWindow('canny Bild')
            cv2.destroyWindow('canny Bild no border')
        # window for sobel image
        cv2.namedWindow('sobel Bild')
        cv2.moveWindow('sobel Bild', 200, 0)
        # add slider for threshold
        cv2.namedWindow('sobel Bild no border')
        cv2.moveWindow('sobel Bild no border', 400, 0)
        cv2.createTrackbar('threshold', 'sobel Bild', 0, 255, threshold_handler_sobel)
        cv2.imshow('sobel Bild', sobel_img*255)
        cv2.imshow('sobel Bild no border', sobel_img_no_border*255)
    else:
        cv2.destroyWindow('sobel Bild')
        cv2.destroyWindow('sobel Bild no border')
        # window for canny image
        cv2.namedWindow('canny Bild')
        cv2.moveWindow('canny Bild', 200, 0)
        cv2.namedWindow('canny Bild no border')
        cv2.moveWindow('canny Bild no border', 400, 0)
        cv2.createTrackbar('threshold', 'canny Bild', 0, 255, threshold_handler_canny)
        cv2.imshow('canny Bild', canny_img*255)
        cv2.imshow('canny Bild no border', canny_img_no_border*255)
sobel_canny_draw(sobel_canny)

# add slider for changing image with noise
cv2.createTrackbar('Noise', 'Originalbild', 0, 10, image_change)
cv2.createTrackbar('Sobel/Canny', 'Originalbild', 0, 1, sobel_canny_draw)

# functions for preprocessing
def sigma_handler(x):
    # define global variables
    global sigma
    global kernel_size_blur
    global preprocessing_algorithm
    global kernel_size
    global img
    global preprocessed_img
    global preprocessed_img_no_border

    # set sigma
    sigma = x
    # apply preprocessing
    preprocessed_img = pp.preprocessing(img, sigma, kernel_size_blur, preprocessing_algorithm, kernel_size)
    preprocessed_img_no_border = pp.preprocessing(img, sigma, kernel_size_blur, preprocessing_algorithm, kernel_size,borderType=cv2.BORDER_ISOLATED)
    # show image
    cv2.imshow('vorverarbeitetes Bild', preprocessed_img)
    cv2.imshow('vorverarbeitetes Bild no border', preprocessed_img_no_border)

    # update sobel and canny
    global threshold_sobel
    global threshold_canny
    if(sobel_canny == 0):
        threshold_handler_sobel(threshold_sobel)
    else:
        threshold_handler_canny(threshold_canny)


def kernel_size_blur_handler(x):
    # define global variables
    global sigma
    global kernel_size_blur
    global preprocessing_algorithm
    global kernel_size
    global img
    global preprocessed_img
    global preprocessed_img_no_border

    
    # set kernel size
    kernel_size_blur = 2 * x + 3
    # apply preprocessing
    preprocessed_img = pp.preprocessing(img, sigma, kernel_size_blur, preprocessing_algorithm, kernel_size)
    preprocessed_img_no_border = pp.preprocessing(img, sigma, kernel_size_blur, preprocessing_algorithm, kernel_size,borderType=cv2.BORDER_ISOLATED)
    # show image
    cv2.imshow('vorverarbeitetes Bild', preprocessed_img)
    cv2.imshow('vorverarbeitetes Bild no border', preprocessed_img_no_border)

    # update sobel and canny
    global threshold_sobel
    global threshold_canny
    if(sobel_canny == 0):
        threshold_handler_sobel(threshold_sobel)
    else:
        threshold_handler_canny(threshold_canny)

def preprocessing_algorithm_handler(x):
    # define global variables
    global sigma
    global kernel_size_blur
    global preprocessing_algorithm
    global kernel_size
    global img
    global preprocessed_img
    global preprocessed_img_no_border
    
    # set kernel size
    preprocessing_algorithm = x
    # apply preprocessing
    preprocessed_img = pp.preprocessing(img, sigma, kernel_size_blur, preprocessing_algorithm, kernel_size)
    preprocessed_img_no_border = pp.preprocessing(img, sigma, kernel_size_blur, preprocessing_algorithm, kernel_size,borderType=cv2.BORDER_ISOLATED)
    # show image
    cv2.imshow('vorverarbeitetes Bild', preprocessed_img)
    cv2.imshow('vorverarbeitetes Bild no border', preprocessed_img_no_border)

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

    # update sobel and canny
    global threshold_sobel
    global threshold_canny
    if(sobel_canny == 0):
        threshold_handler_sobel(threshold_sobel)
    else:
        threshold_handler_canny(threshold_canny)

def kernel_size_handler(x):
    # define global variables
    global sigma
    global kernel_size_blur
    global preprocessing_algorithm
    global kernel_size
    global img
    global preprocessed_img
    global preprocessed_img_no_border
    
    # set kernel size
    kernel_size = 2 * x + 3
    # apply preprocessing
    preprocessed_img = pp.preprocessing(img, sigma, kernel_size_blur, preprocessing_algorithm, kernel_size)
    preprocessed_img_no_border = pp.preprocessing(img, sigma, kernel_size_blur, preprocessing_algorithm, kernel_size,borderType=cv2.BORDER_ISOLATED)
    # show image
    cv2.imshow('vorverarbeitetes Bild', preprocessed_img)
    cv2.imshow('vorverarbeitetes Bild no border', preprocessed_img_no_border)

    # update sobel and canny
    global threshold_sobel
    global threshold_canny
    if(sobel_canny == 0):
        threshold_handler_sobel(threshold_sobel)
    else:
        threshold_handler_canny(threshold_canny)

# add sliders for preprocessing
cv2.createTrackbar('sigma', 'vorverarbeitetes Bild', 0, 10, sigma_handler)
cv2.createTrackbar('kernel_size_blur', 'vorverarbeitetes Bild', 0, 10, kernel_size_blur_handler)
cv2.createTrackbar('preprocessing_algorithm', 'vorverarbeitetes Bild', 0, 5, preprocessing_algorithm_handler)
cv2.createTrackbar('kernel_size', 'vorverarbeitetes Bild', 0, 3, kernel_size_handler)

# canny image:
# check opencv documentation for thresholds: 
# everything below low_threshold is not an edge
# everything above high_threshold is an edge
# everything between low_threshold and high_threshold is an 
# edge if it is connected to an edge above high_threshold

# slider for low_threshold
# slider for high_threshold
# equalize thresholds

# XOR image:
# output for sum of pixels (error value)

# window for XOR image
cv2.namedWindow('XOR Bild')
cv2.moveWindow('XOR Bild', 400, 0)

XOR_img = np.bitwise_xor(sobel_img, sobel_img_no_border)

# calculate XOR image with numpy
def XOR_img_handler():
    global XOR_img
    global sobel_img
    global sobel_img_no_border
    global canny_img
    global canny_img_no_border
    global sobel_canny

    if(sobel_canny == 0):
        XOR_img = np.bitwise_xor(sobel_img, sobel_img_no_border)
        print(sobel_img[0])
        print(sobel_img_no_border[0])
        print(np.sum(XOR_img[0]))
        cv2.imshow('XOR Bild', XOR_img*255)
    else:
        XOR_img = np.bitwise_xor(canny_img, canny_img_no_border)
        cv2.imshow('XOR Bild', XOR_img*255)
    # print error value
    error_value = np.sum(XOR_img)
    print("XOR Error Value: " + str(error_value))    

# show image
cv2.imshow('XOR Bild', XOR_img*255)

# start program
cv2.waitKey(0)
cv2.destroyAllWindows()