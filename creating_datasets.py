# the datasets were obtained from Kaggle.com. Make sure to download these datasets and store them in your working directory. They will be downloaded as "Training/", "Test/", and "test-multiple_fruits" the specific url is: https://www.kaggle.com/moltean/fruits. At the time of obtaining the dataset, there were only 65k images -- still plenty of images to work with.

# This is part 1 of the entire process, hence manually creating the training and testing sets by converting the images to pixel form (3D matrices -- 100x100x3)

# Creating Training and Testing datasets

# over 65k total images, 95 different fruits


import numpy as np

from PIL import Image

import glob
import os

import pickle



############################################################################################

# loop through all files. loop through all images. read images --> get pixels --> store pixels
# as dict --> append dict to list

# FOR TRIANING

all_fruit_files_train = []

for fruit_file in glob.glob("Training/*"):		# make sure you are working in the directory where the "Training/" folder is located. The same goes for "Test/" below.
	all_fruit_files_train.append(fruit_file) 	# appending all fruit folders to one list



# store pixel values as dict

IMAGES_TRAINING = []

counter = 0

class_number = np.arange(len(all_fruit_files_train))


for i in all_fruit_files_train:

	images_file = glob.glob(i + '/*.jpg') 	# read each image file
	clss = class_number[counter]
	counter += 1

	for j in images_file:
		im = Image.open(j)
		w, h = im.size 	# get width and height of image
		pixels = list(im.getdata()) 	# get pixel data as list
		pixels = np.array(pixels).reshape(w, h, 3) # convert to array with dimensions w x h x 3
		dict = { clss : pixels } # create dict
		IMAGES_TRAINING.append(dict) # store dict




# FOR TESTING


IMAGES_TESTING = []

all_fruit_files_test = []

for fruit_file in glob.glob("Test/*"):
	all_fruit_files_test.append(fruit_file)



# making sure that glob does not mess up the classes.
# Class = 0 should be Strawberry wedge and class = -1 (last class) should be Pomelo Sweetie
a1 = []
a2 = []

s1 = len('Training/')
s2 = len('Test/')

for i in all_fruit_files_train:
	a1.append(i[s1:])
for i in all_fruit_files_test:
	a2.append(i[s2:])

if a1 == a2:
	print('\nclasses are the same for training and testing sets, glob does not mess it up')
	print('can do print(a1) and print(a2) to check\n')

# defining classes
classes = a1

counter = 0

class_number = np.arange(len(all_fruit_files_test))

# store pixel values as dict

for i in all_fruit_files_test:

	images_files = glob.glob(i + '/*.jpg')
	clss = class_number[counter]
	counter += 1

	for j in images_file:
		im = Image.open(j)
		w, h = im.size
		pixels = list(im.getdata())
		pixels = np.array(pixels).reshape(w, h, 3)
		dict = { clss : pixels }
		IMAGES_TESTING.append(dict)




# to check, you can run these commands:
#
# len(IMAGES_TRAINING) 			# output: total number of training images
# len(IMAGES_TRAINING[0])		# output: 1 (because this is one dict value, or image)
# IMAGES_TRAINING[0][0].shape	# output: (100, 100, 3), here the first [0] is the image, and the second [0]
								# is the class. [737][0] is the last image for Strawberry Wedge, or class 0



###################

# pickling

with open('images_training.pkl', 'wb') as f:
	pickle.dump(IMAGES_TRAINING, f)
with open('images_testing.pkl', 'wb') as f:
	pickle.dump(IMAGES_TESTING, f)

with open('classes.pkl', 'wb') as f:
	pickle.dump(classes, f)



"""
# to load:
with open('images_training.pkl', 'rb') as f:
	images_training = pickle.load(f)
"""
