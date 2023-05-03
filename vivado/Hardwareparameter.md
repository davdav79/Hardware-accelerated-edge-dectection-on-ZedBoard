## 1st implementiation
### basic requirements
- only sobel and canny algorithm without preprocessing
- edge problem is solved by discard pixels at the edge of the image
- kernel size is fixed to 3x3 for both sobel and canny

### coprocessor procedure
1. image is loaded by coprocessor
2. coprocessor rescales the image to a resonable size (512x512)
3. coprocessor converts image to grayscale 8bit (uint8_t)
4. coprocessor divides image into kernel sized blocks
5. coprocessor sends blocks and kernel parameters to FPGA
6. FPGA applies convolution and sends result back to coprocessor
7. coprocessor saves result as separate image

### Sequence in the FPGA (sobel)
1. FPGA receives image block, kernel parameters and threshold
2. FPGA applies convolution for x and y kernel
3. FPGA calculates magnitude 
4. FPGA binarizes magnitude with threshold
5. FPGA sends result back to coprocessor

### Sequence in the FPGA (canny)
1. FPGA calculates magnitude with sobel sequence
2. FPGA calculates angle
3. ...




# in theory:

## coprocessor preprocessing (not hardware accelerated):
1. rescale image to reasonable size (512x512)
2. convert image to grayscale 8bit (uint8_t)
3. define max. kernel size for convolution (3x3)

## preprocess image
### edge extension (hardware accelerated?):
1. coprocessor extends image by kernel size (kernel size + size of gaussian blur kernel to blur edge extension) / FPGA calculates edge values (depending on max. kernel size)

### gaussian blur (hardware accelerated):
1. coprocessor divides image into kernel sized blocks
2. coprocessor sends blocks and kernel parameters to FPGA
3. FPGA applies convolution and sends result back to coprocessor

## apply algorithm
### sobel x filter (hardware accelerated):
1. coprocessor sends image blocks and parameters for x kernel back to FPGA
2. FPGA applies convolution and sends result back to coprocessor
3. coprocessor saves result as separate image

### sobel y filter (hardware accelerated):
1. coprocessor sends image blocks and parameters for y kernel back to FPGA
2. FPGA applies convolution and sends result back to coprocessor
3. coprocessor saves result as separate image

### combine sobel x and y filter and apply threshold (hardware accelerated):
1. coprocessor sends image blocks x and y and threshold value to FPGA
2. FPGA calculates magnitude (and angle for canny)
3. FPGA binarizes magnitude with received threshold
4. FPGA sends result back to coprocessor

# TODO: erstmal Sobel und Canny und später erst Preprocessing dh. Bild erst croppen und nicht blurren


