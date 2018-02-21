import sys
from imageReader import *
from psnr import *
from PIL import Image, ImageOps
from ssim import SSIM
from ssim.utils import get_gaussian_kernel

gaussian_kernel_sigma = 1.5
gaussian_kernel_width = 11
gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)

if __name__ == '__main__':

    original = "../dataset/train_orig/orig_1.png"
    encoded = "../dataset/train_mf/mf_1.png"

    original_width, original_height, original_npix = get_image_data(original)
    encoded_width, encoded_height, encoded_npix = get_image_data(original)

    if original_width != encoded_width or original_height != encoded_height or original_npix != encoded_npix:
        print("ERROR: Images should have same dimensions. \n")
        exit(1)

    original_y, original_cb, original_cr = get_yuv(original)
    encoded_y, encoded_cb, encoded_cr = get_yuv(encoded)
    calculate_psnr(original_y, encoded_y, original_cb, encoded_cb, original_cr, encoded_cr, original_npix)

    size = (256,256)
    im = Image.open(original)
    im = im.resize(size, Image.ANTIALIAS)
    mf = Image.open(encoded)
    mf = mf.resize(size, Image.ANTIALIAS)
    ssim = SSIM(im, gaussian_kernel_1d).ssim_value(mf)
    print("SSIM={0}".format(ssim))