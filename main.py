import tensorflow as tf
import os
from PIL import Image
import numpy as np

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

def res_unit(z,in_filter):
	z_prev=z
	if in_filter==2:
		z=_conv(z,[3,3,2,64],1)
	else:
		z=_conv(z,[3,3,64,64],1)
	z=_batch_norm(z)
	z=lrelu(z,0.2)
	z=_conv(z,[3,3,64,64],1)
	z=_batch_norm(z)
	if in_filter!=2:
		z=z+z_prev
	z=lrelu(z,0.2)
	return z

def gen(z,reuse):
	with tf.variable_scope('generator',reuse=reuse):
		z_init=z
		z=res_unit(z,2)
		for i in range(7):
			z=res_unit(z,64)
		z=_conv(z,[3,3,64,2],1)
		z=z+z_init
		z=tf.nn.tanh(z)
		return z	

def disc_unit(x,depth,stride,in_filter):
	x=_conv(x,[3,3,in_filter,depth],stride)
	x=_batch_norm(x)
	x=lrelu(x,0.2)
	return x

def disc(x,reuse):
	with tf.variable_scope('discriminator',reuse=reuse):
		x=disc_unit(x,16,1,2)
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

if __name__=='__main__':
	tf.reset_default_graph()
	orig=Image.open("../dataset/cropped_train/crop_1.png")
	orig=np.array(orig)
	orig=orig.reshape(-1,512,512,2)
	mf=Image.open("../dataset/cropped_train_mf/mf_crop_1.png")
	mf=np.array(mf)
	mf=mf.reshape(-1,512,512,2)
	z=tf.placeholder(tf.float32,[None,512,512,2])
	x=tf.placeholder(tf.float32,[None,512,512,2])
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
	with tf.Session() as sess:
		sess.run(init)
		dloss=sess.run(d_loss,feed_dict={x:orig,z:mf})
		gloss=sess.run(g_loss,feed_dict={z:mf})
		print(dloss)
		print(gloss)
		
		

