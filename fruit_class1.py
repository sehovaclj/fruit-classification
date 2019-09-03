# for self learning, from square 0 (full image) to full image classification model

# over 65k total images, 95 different fruits


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init

import random
import time

import math

from PIL import Image

import glob
import os




############################################################################################

# loop through all files. loop through all images. read images --> get pixels --> store pixels
# as dict --> append dict to massive list

# FOR TRIANING

all_fruit_files_train = []

for fruit_file in glob.glob("Training/*"):
	all_fruit_files_train.append(fruit_file)


# store pixel values as dict

IMAGES_TRAINING = []

for i in range(len(all_fruit_files_train)):
	images_file = glob.glob(all_fruit_files_train[i] + '/*.jpg')
	for j in images_file:
		im = Image.open(j)
		w, h = im.size
		pixels = list(im.getdata())
		pixels = np.array(pixels).reshape(w, h, 3)
		dict = { i : pixels }
		IMAGES_TRAINING.append(dict)




# FOR TESTING


IMAGES_TESTING = []


all_fruit_files_test = []

for fruit_file in glob.glob("Test/*"):
	all_fruit_files_test.append(fruit_file)


# store pixel values as dict

for i in range(len(all_fruit_files_test)):
	images_files = glob.glob(all_fruit_files_test[i] + '/*.jpg')
	for j in images_file:
	im = Image.open(j)
	w, h = im.size
	pixels = list(im.getdata())
	pixels = np.array(pixels).reshape(w, h, 3)
	dict = { i : pixels }
	IMAGES_TESTING.append(dict)









