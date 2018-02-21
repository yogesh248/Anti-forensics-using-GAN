import os
import numpy as np
import rawpy
from PIL import Image
from PIL import ImageFilter

'''
#number the images
ctr=1
for file in os.listdir("../dataset/original"):
	os.rename("../dataset/original/"+file,"../dataset/original/orig_"+str(ctr)+".NEF")
	ctr=ctr+1
'''
'''
for i in range(788,1001):
	raw=rawpy.imread("../dataset/original/orig_"+str(i)+".NEF")
	rgb=raw.postprocess()
	image=Image.fromarray(rgb)
	gray=image.convert('LA')
	gray.save("../dataset/grayscale/gray_"+str(i)+".png")
'''
'''
#convert to grayscale
ctr=1
for file in os.listdir("../dataset/original"):
	raw=rawpy.imread("../dataset/original/"+file)
	rgb=raw.postprocess()
	image=Image.fromarray(rgb)
	gray=image.convert('LA')
	gray.save("../dataset/grayscale/gray_"+str(ctr)+".png")
	ctr=ctr+1
'''
'''
#crop images
ctr=1
for file in os.listdir("../dataset/grayscale"):
	image=Image.open("../dataset/grayscale/"+file)
	width,height=image.size
	crop_size=512
	r1=l4=l2=r3=int(width/2)
	b1=t3=t4=b2=int(height/2)
	r2=r4=int(width/2)+crop_size
	l1=l3=int(width/2)-crop_size
	b3=b4=int(height/2)+crop_size
	t1=t2=int(height/2)-crop_size
	crop1=image.crop((l1,t1,r1,b1))
	crop2=image.crop((l2,t2,r2,b2))
	crop3=image.crop((l3,t3,r3,b3))
	crop4=image.crop((l4,t4,r4,b4))
	crop1.save("../dataset/cropped/crop_"+str(ctr)+".png")
	ctr=ctr+1
	crop2.save("../dataset/cropped/crop_"+str(ctr)+".png")
	ctr=ctr+1
	crop3.save("../dataset/cropped/crop_"+str(ctr)+".png")
	ctr=ctr+1
	crop4.save("../dataset/cropped/crop_"+str(ctr)+".png")
	ctr=ctr+1
'''


#median filter
'''
for i in range(1,3201):
	image=Image.open("../dataset/cropped_train/crop_"+str(i)+".png")
	med=image.filter(ImageFilter.MedianFilter(size=3))
	med.save("../dataset/cropped_train_mf/mf_crop_"+str(i)+".png")	
for i in range(3201,4001):
	image=Image.open("../dataset/cropped_test/crop_"+str(i)+".png")
	med=image.filter(ImageFilter.MedianFilter(size=3))
	med.save("../dataset/cropped_test_mf/mf_crop_"+str(i)+".png")	
'''

'''
os.chdir("../dataset/train_orig")
for i in range(1,3201):
	os.rename("crop_"+str(i)+".png","orig_"+str(i)+".png")
os.chdir("../train_mf")
for i in range(1,3201):	
	os.rename("mf_crop_"+str(i)+".png","mf_"+str(i)+".png")
ctr=1	
os.chdir("../test_orig")
for i in range(3201,4001):
	os.rename("crop_"+str(i)+".png","orig_"+str(ctr)+".png")
	ctr=ctr+1
ctr=1	
os.chdir("../test_mf")
for i in range(3201,4001):	
	os.rename("mf_crop_"+str(i)+".png","mf_"+str(ctr)+".png")
	ctr=ctr+1
'''	