import tensorflow as tf
import os
from PIL import Image
import numpy as np
import cv2

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

if __name__=='__main__':
	batch_size=2
	num_epochs=1
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
	with tf.Session() as sess:
		sess.run(init)
		for i in range(num_epochs):
			step=1
			for j in range(10):
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
		saver.save(sess,"../temp/ckpt/model.ckpt")		
		
		

