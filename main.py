import tensorflow as tf
import os
import numpy as np
import cv2
import sys
import math
from PIL import Image, ImageOps
from ssim import SSIM
from ssim.utils import get_gaussian_kernel

filter_1=np.array([[-0.0833,0.1667,-0.1667,0.1667,-0.0833],
	[0.1667,-0.5,0.6667,-0.5,0.1667],
	[-0.1667,0.6667,-1.0,0.6667,-0.1667],
	[0.1667,-0.5,0.6667,-0.5,0.1667],
	[-0.0833,0.1667,-0.1667,0.1667,-0.0833]])
filter_1=filter_1.reshape((5,5,1,1))

filter_2=np.array([[0,0,0.0199,0,0],
	[0,0.0897,0.1395,0.0897,0],
	[-0.199,0.1395,-1.0,0.1395,0.0199],
	[0,0.0897,0.1395,0.0897,0],
	[0,0,0.0199,0,0]])
filter_2=filter_2.reshape((5,5,1,1))

filter_3=np.array([[0.0562,-0.1354,0,0.1354,-0.0562],
	[0.0818,-0.197,0,0.197,-0.0818],
	[0.0926,-0.2233,0,0.2233,-0.0926],
	[0.0818,-0.197,0,0.197,-0.0818],
	[0.0562,-0.1354,0,0.1354,-0.0562]])
filter_3=filter_3.reshape((5,5,1,1))

filter_4=np.array([[-0.0562,-0.0818,-0.0926,-0.0818,-0.0562],
	[0.1354,0.197,0.2233,0.197,0.1354],
	[0,0,0,0,0],
	[-0.1354,-0.197,-0.2233,-0.197,-0.1354],
	[0.0562,0.0818,0.0926,0.0818,0.0562]])
filter_4=filter_4.reshape((5,5,1,1))

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
    return psnr_y

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

def get_rgb(im,npix):    
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

def get_yuv(im):
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

def init_tensor(shape):
	return tf.Variable(tf.truncated_normal(shape, mean=0.0, stddev=1.0))

def lrelu(x, alpha):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def _batch_norm(input_):
	assert len(input_.get_shape()) == 4
	filter_shape = input_.get_shape().as_list()
	mean, var = tf.nn.moments(input_, axes=[0, 1, 2])
	out_channels = filter_shape[3]
	offset = tf.Variable(tf.zeros([out_channels]))
	scale = tf.Variable(tf.ones([out_channels]))
	batch_norm = tf.nn.batch_normalization(input_, mean, var, offset, scale, 0.001)
	return batch_norm

def _conv(input, filter_shape, stride):
	return tf.nn.conv2d(input,filter=init_tensor(filter_shape),strides=[1, stride, stride, 1],padding="SAME")

def gen_unit(z,in_filter):
	z_prev=z
	if in_filter==1:
		z=_conv(z,[3,3,1,64],1)
	else:
		z=_conv(z,[3,3,64,64],1)
	z=_batch_norm(z)
	z=lrelu(z,0.2)
	z=_conv(z,[3,3,64,64],1)
	z=_batch_norm(z)
	if in_filter!=1:
		z=z+z_prev
	z=lrelu(z,0.2)
	return z

def gen(z,reuse):
	with tf.variable_scope('generator',reuse=reuse):
		z_init=z
		z=gen_unit(z,1)
		for i in range(7):
			z=gen_unit(z,64)
		z=_conv(z,[3,3,64,1],1)
		z=z+z_init
		z=tf.nn.tanh(z)
		return z	

def disc_unit(x,depth,stride,in_filter):
	x=_conv(x,[3,3,in_filter,depth],stride)
	x=_batch_norm(x)
	x=lrelu(x,0.2)
	return x

def disc(x,reuse):
	x=tf.nn.conv2d(x,filter=filter_1,strides=[1,1,1,1],padding="SAME")
	x=tf.nn.conv2d(x,filter=filter_2,strides=[1,1,1,1],padding="SAME")
	x=tf.nn.conv2d(x,filter=filter_3,strides=[1,1,1,1],padding="SAME")
	x=tf.nn.conv2d(x,filter=filter_4,strides=[1,1,1,1],padding="SAME")
	with tf.variable_scope('discriminator',reuse=reuse):
		x=disc_unit(x,16,1,1)
		x=disc_unit(x,16,2,16)
		x=disc_unit(x,32,1,16)
		x=disc_unit(x,32,2,32)
		x=disc_unit(x,64,1,32)
		x=disc_unit(x,64,2,64)
		x=disc_unit(x,128,1,64)
		x=disc_unit(x,128,2,128)
		x=tf.contrib.layers.fully_connected(tf.contrib.layers.flatten(x),256)
		x=lrelu(x,0.2)
		x=tf.contrib.layers.fully_connected(x,1)
		x=tf.nn.sigmoid(x)
		return x

def next_batch_orig(step,batch_size):
	arr=np.empty([batch_size,128,128,1])
	j=0
	for i in range(step,step+batch_size):
		orig=Image.open("../dataset/train_orig/orig_{0}.png".format(i))
		orig=np.array(orig)
		orig=orig.reshape((128,128,1))
		arr[j]=orig
		j=j+1
	return arr	

def next_batch_mf(step,batch_size):
	arr=np.empty([batch_size,128,128,1])
	j=0
	for i in range(step,step+batch_size):
		mf=Image.open("../dataset/train_mf/mf_{0}.png".format(i))
		mf=np.array(mf)
		mf=mf.reshape((128,128,1))
		arr[j]=mf
		j=j+1
	return arr	

train_size=3200
test_size=800
tf.reset_default_graph()
z=tf.placeholder(tf.float32,[None,128,128,1])
x=tf.placeholder(tf.float32,[None,128,128,1])
rest=gen(z,reuse=False)
d_orig=disc(x,reuse=False)
d_rest=disc(rest,reuse=True)
d_loss=-tf.reduce_mean(tf.log(d_orig)+tf.log(1.-d_rest))
g_loss=-tf.reduce_mean(tf.log(d_rest))
tvars=tf.trainable_variables()
d_vars=[var for var in tvars if var.name.startswith('discriminator')]
g_vars=[var for var in tvars if var.name.startswith('generator')]
d_train=tf.train.AdamOptimizer(learning_rate=5e-6,beta1=0.9,beta2=0.999,epsilon=1e-8).minimize(d_loss,var_list=d_vars)
g_train=tf.train.AdamOptimizer(learning_rate=5e-4,beta1=0.9,beta2=0.999,epsilon=1e-8).minimize(g_loss,var_list=g_vars)
init=tf.global_variables_initializer()
saver=tf.train.Saver()


def train(batch_size,num_epochs):
	os.system("rm -rf ../temp/gen/*")
	os.system("rm -rf ../temp/ckpt/*")
	with tf.Session() as sess:
		sess.run(init)
		for i in range(num_epochs):
			step=1
			for j in range(int(train_size/batch_size)):
				X=next_batch_orig(step,batch_size)
				Z=next_batch_mf(step,batch_size)
				step=step+batch_size
				_,dloss=sess.run([d_train,d_loss],feed_dict={x:X,z:Z})
				_,gloss=sess.run([g_train,g_loss],feed_dict={z:Z})
				print("Iteration {0} Dloss={1} Gloss={2}".format(j,dloss,gloss))
				Z=Image.open("../dataset/test_mf/mf_{0}.png".format(j+1))
				Z=np.array(Z)
				Z=Z.reshape((1,128,128,1))
				image=sess.run(rest,feed_dict={z:Z})
				image=image.reshape((128,128))
				image=Image.fromarray(image,mode='L')
				image.save("../temp/gen/gen_{0}.png".format(j+1))
			saver.save(sess,"../temp/ckpt/model.ckpt",global_step=i+1)		

def test():
	tf.get_default_graph()
	saver=tf.train.import_meta_graph("../temp/ckpt/model.ckpt-1.meta")
	with tf.Session() as sess:
		saver.restore(sess,tf.train.latest_checkpoint("../temp/ckpt/"))
		for i in range(1,test_size+1):
			print("Iteration {0}".format(i))
			original_image="../dataset/test_orig/orig_{0}.png".format(i)
			mf_image="../dataset/test_mf/mf_{0}.png".format(i)
			original_image_object=Image.open(original_image)
			mf_image_object=Image.open(mf_image)
			Z=np.array(mf_image_object)
			Z=Z.reshape((1,128,128,1))
			image=sess.run(rest,feed_dict={z:Z})
			image=image.reshape((128,128))
			gen_image_object=Image.fromarray(image,mode='L')
			original_width, original_height, original_npix = get_image_data_from_file(original_image)
			gen_width, gen_height, gen_npix = get_image_data(gen_image_object)
			mf_width, mf_height, mf_npix = get_image_data_from_file(mf_image)
			if original_width != gen_width or original_height != gen_height or original_npix != gen_npix:
				print("ERROR: Images should have same dimensions. \n")
				exit(1)
			if mf_width != gen_width or mf_height != gen_height or mf_npix != gen_npix:
				print("ERROR: Images should have same dimensions. \n")
				exit(1)	
			original_y, original_cb, original_cr = get_yuv(original_image_object)
			gen_y, gen_cb, gen_cr = get_yuv(gen_image_object)
			mf_y, mf_cb, mf_cr = get_yuv(mf_image_object)
			psnr=calculate_psnr(original_y, gen_y, original_cb, gen_cb, original_cr, gen_cr, original_npix)
			print("oPSNR={0}".format(psnr))
			psnr=calculate_psnr(mf_y, gen_y, mf_cb, gen_cb, mf_cr, gen_cr, mf_npix)
			print("mPSNR={0}".format(psnr))
			size = (256,256)
			original_image_object = original_image_object.resize(size, Image.ANTIALIAS)
			gen_image_object = gen_image_object.resize(size, Image.ANTIALIAS)
			mf_image_object = mf_image_object.resize(size, Image.ANTIALIAS)
			ssim = SSIM(original_image_object, gaussian_kernel_1d).ssim_value(gen_image_object)
			print("oSSIM={0}".format(ssim))
			ssim = SSIM(mf_image_object, gaussian_kernel_1d).ssim_value(gen_image_object)
			print("mSSIM={0}".format(ssim))

if __name__=='__main__':
	batch_size=16
	num_epochs=1
	train(batch_size,num_epochs)
	test()	
		

