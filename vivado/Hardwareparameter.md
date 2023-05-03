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
3. FPGA sends result back to coprocessor

# TODO: erstmal Sobal und Canny und später erst PReprocessing dh. Bild erst croppen und nicht blurren

