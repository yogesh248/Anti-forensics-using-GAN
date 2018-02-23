import os
import numpy as np
import rawpy
from PIL import Image
from PIL import ImageFilter
import cv2


#convert to grayscale
for i in range(1,1001):
	raw=rawpy.imread("E:/project/original/orig_"+str(i)+".NEF")
	rgb=raw.postprocess()
	image=Image.fromarray(rgb)
	gray=image.convert('L')
	gray.save("E:/project/grayscale/gray_"+str(i)+".png")

#crop images
ctr=1
crop_size=512
size=128,128
for file in os.listdir("E:/project/grayscale"):
	image=Image.open("E:/project/grayscale/"+file)
	width,height=image.size
	r1=l4=l2=r3=int(width/2)
	b1=t3=t4=b2=int(height/2)
	r2=r4=int(width/2)+crop_size
	l1=l3=int(width/2)-crop_size
	b3=b4=int(height/2)+crop_size
	t1=t2=int(height/2)-crop_size
	crop1=image.crop((l1,t1,r1,b1))
	crop1.thumbnail(size,Image.ANTIALIAS)
	crop2=image.crop((l2,t2,r2,b2))
	crop2.thumbnail(size,Image.ANTIALIAS)
	crop3=image.crop((l3,t3,r3,b3))
	crop3.thumbnail(size,Image.ANTIALIAS)
	crop4=image.crop((l4,t4,r4,b4))
	crop4.thumbnail(size,Image.ANTIALIAS)
	crop1.save("E:/project/cropped/crop_"+str(ctr)+".png")
	ctr=ctr+1
	crop2.save("E:/project/cropped/crop_"+str(ctr)+".png")
	ctr=ctr+1
	crop3.save("E:/project/cropped/crop_"+str(ctr)+".png")
	ctr=ctr+1
	crop4.save("E:/project/cropped/crop_"+str(ctr)+".png")
	ctr=ctr+1

for i in range(1,3201):
	os.system("mv E:/project/cropped/crop_{0}.png E:/project/train_orig/orig_{1}.png".format(i,i))
for i in range(3201,4001):
	os.system("mv E:/project/cropped/crop_{0}.png E:/project/test_orig/orig_{1}.png".format(i,i-3200))	

#median filter
for i in range(1,3201):
	image=Image.open("E:/project/train_orig/orig_"+str(i)+".png")
	med=image.filter(ImageFilter.MedianFilter(size=3))
	med.save("E:/project/train_mf/mf_"+str(i)+".png")	
for i in range(3201,4001):
	image=Image.open("E:/project/test_orig/orig_"+str(i-3200)+".png")
	med=image.filter(ImageFilter.MedianFilter(size=3))
	med.save("E:/project/test_mf/mf_"+str(i-3200)+".png")	
