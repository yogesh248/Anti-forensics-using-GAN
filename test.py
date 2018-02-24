import sys
import math
from PIL import Image, ImageOps
from ssim import SSIM
from ssim.utils import get_gaussian_kernel
import tensorflow as tf
import numpy as np

gaussian_kernel_sigma = 1.5
gaussian_kernel_width = 11
gaussian_kernel_1d = get_gaussian_kernel(gaussian_kernel_width, gaussian_kernel_sigma)

def calculate_psnr(original_y, encoded_y, original_cb, encoded_cb, original_cr, encoded_cr, npix):
    error_y = 0  
    error_cb = 0  
    error_cr = 0
    for i in range(0, len(original_y)):
        dif_y = abs(original_y[i] - encoded_y[i])  
        dif_cb = abs(original_cb[i] - encoded_cb[i])  
        dif_cr = abs(original_cr[i] - encoded_cr[i])  
        error_y += dif_y * dif_y  
        error_cb += dif_cb * dif_cb  
        error_cr += dif_cr * dif_cr  
    mse_y = float(error_y) / float(npix)  
    mse_cb = float(error_cb) / float(npix)  
    mse_cr = float(error_cr) / float(npix)  
    if mse_y != 0:
        psnr_y = float(-10.0 * math.log(mse_y / (255 * 255), 10))
    else:
        psnr_y = 0
    if mse_cb != 0:
        psnr_cb = float(-10.0 * math.log(mse_cb / (255 * 255), 10))
    else:
        psnr_cb = 0
    if mse_cr != 0:
        psnr_cr = float(-10.0 * math.log(mse_cr / (255 * 255), 10))
    else:
        psnr_cr = 0
    print("PSNR={0}".format(psnr_y))

def get_image_data_from_file(filename):
    im = Image.open(filename)
    width = im.size[0]
    height = im.size[1]
    npix = im.size[0] * im.size[1]
    return width, height, npix

def get_image_data(im):
    width = im.size[0]
    height = im.size[1]
    npix = im.size[0] * im.size[1]
    return width, height, npix    

def get_rgb(filename, npix):    
    im = Image.open(filename)
    rgb_im = im.convert('RGB')
    r = [-1] * npix
    g = [-1] * npix
    b = [-1] * npix
    for y in range(0, im.size[1]):
        for x in range(0, im.size[0]):
            rpix, gpix, bpix = rgb_im.getpixel((x, y))
            r[im.size[0] * y + x] = rpix
            g[im.size[0] * y + x] = gpix
            b[im.size[0] * y + x] = bpix
    return r, g, b

def get_yuv(filename):
    im = Image.open(filename)
    im = im.convert('YCbCr')
    y = []
    u = []
    v = []
    for pix in list(im.getdata()):
        y.append(pix[0])
        u.append(pix[1])
        v.append(pix[2])
    return y, u, v

def rgb_to_yuv(r, g, b):  
    y = [0] * len(r)
    cb = [0] * len(r)
    cr = [0] * len(r)
    for i in range(0, len(r)):
        y[i] = int(0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i])
        cb[i] = int(128 - 0.168736 * r[i] - 0.331364 * g[i] + 0.5 * b[i])
        cr[i] = int(128 + 0.5 * r[i] - 0.418688 * g[i] - 0.081312 * b[i])
    return y, cb, cr

if __name__ == '__main__':
    with tf.Session() as sess:
        saver=tf.train.import_meta_graph("../temp/ckpt/model.ckpt.meta")
        saver.restore(sess,tf.train.latest_checkpoint("../temp/ckpt/"))
        graph=tf.get_default_graph()
        z=graph.get_tensor_by_name("z:0")
        rest=graph.get_tensor_by_name("rest:0")
        for i in range(1,11):
            print("Iteration {0}".format(i))
            original="../dataset/test_orig/orig_{0}.png".format(i)
            mf="../dataset/test_mf/mf_{0}.png".format(i)
            mf_i=Image.open(mf)
            Z=np.array(mf_i)
            Z=Z.reshape((1,128,128,1))
            image=sess.run(rest,feed_dict={z:Z})
            image=image.reshape((128,128))
            gen=Image.fromarray(image,mode='L')
            print("With respect to original:")
            original_width, original_height, original_npix = get_image_data_from_file(original)
            encoded_width, encoded_height, encoded_npix = get_image_data(gen)
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
            print("With respect to MF:")
            original_width, original_height, original_npix = get_image_data_from_file(mf)
            encoded_width, encoded_height, encoded_npix = get_image_data(gen)
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