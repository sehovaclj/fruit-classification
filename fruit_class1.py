# python code that:
# 	loads the training and testing pickle files from 
#	splits the test set into a validation set and final test set
#	contains CNN model to be used for fruit classification
#	training, validation, and update of parameters


# importing functions

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

import pickle



############################################################################################

# building the CNN model

class CNN_Net(nn.Module):
	def __init__(self, C_in, num_filters1, k1, maxpool_k1, maxpool_s1,
		num_filters2, k2, num_filters3, k3, fc_size1, fc_size2, fc_output):
		super(CNN_Net, self).__init__()

		# conv and pooling layers
		self.conv1 = nn.Conv2d(C_in, num_filters1, k1) # first conv layer. Assume square kernels and stride=1. 
		self.hw1 = 100-k1+1 # keeping track of output dimensions for the first fc layer

		self.pool = nn.MaxPool2d(maxpool_k1, maxpool_s1) # max pooling layer. This layer will be the same for each conv layer to keep consistency
		self.hw2 = ((self.hw1-(maxpool_k1-1)-1)/maxpool_s1) + 1

		# check to see if hw2 is a float or an integer
		if not (self.hw2).is_integer():
			print("\nhw2 is not an int, please choose a more appropriate k1, maxpool_k1, and maxpool_s1") 

		self.conv2 = nn.Conv2d(num_filters1, num_filters2, k2) # second conv layer.	

		self.hw3 = self.hw2-k2+1
		self.hw4 = ((self.hw3-(maxpool_k1-1)-1)/maxpool_s1) + 1 # calculating the h and w for the pooling that comes after the second layer
		if not (self.hw4).is_integer():
			print("\nhw4 is not an int, please choose a more appropriate k2, maxpool_k1, and maxpool_s1") 

		self.conv3 = nn.Conv2d(num_filters2, num_filters3, k3) # third conv layer. Refer to pytorch docs for correct math regarding output dimensions
		self.hw5 = self.hw4-k3+1
		self.hw6 = ((self.hw5-(maxpool_k1-1)-1)/maxpool_s1) + 1 

		if not (self.hw6).is_integer():
			print("\nhw6 is not an int, please choose a more appropriate k3, maxpool_k1, and maxpool_s1") 


		self.num_filters3 = num_filters3

		# fully connected layers
		self.fc1 = nn.Linear(int(self.num_filters3*self.hw6*self.hw6), fc_size1)
		self.fc2 = nn.Linear(fc_size1, fc_size2)
		self.fc3 = nn.Linear(fc_size2, fc_output) # output layer will be number of classes = 95



	# forward function, passing images through CNN_Net model
	# note that the kernel sizes (k1, k2, k3) and maxpool kernel and stride (maxpool_k1 and maxpool_s1) are important. These need to coincide with H and W of each conv/maxpool output. Hence the reasoning for the if statements above.
	def forward(self, x):
		x = self.pool(torch.relu(self.conv1(x))) # pass through first conv layer + max pool
		x = self.pool(torch.relu(self.conv2(x))) # pass through second conv layer + max pool
		x = self.pool(torch.relu(self.conv3(x))) # pass through third conv layer + max pool
		x = x.view(-1, int(self.num_filters3*self.hw6*self.hw6)) # change shape to be passed through first fc layer
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = self.fc3(x)

		return x













###################################################################################################

# the main function

def main(seed, cuda, C_in, nf1, k1, mpk1, mps1, nf2, k2, nf3, k3, fc1, fc2, fco):


	# seed == given seed
	np.random.seed(seed)
	torch.manual_seed(seed)


	"""

	##########################################################################################
	# first thing we have to do is load the training and testing images from the pickle files

	# load pickle files
	with open('images_training.pkl', 'rb') as f:
		images_training = pickle.load(f)

	with open('images_testing.pkl', 'rb') as f:
		images_testing = pickle.load(f)

	with open('classes.pkl', 'rb') as f:
		classes = pickle.load(f)

	"""


	# calling the model
	model = CNN_Net(C_in, nf1, k1, mpk1, mps1, nf2, k2, nf3, k3, fc1, fc2, fco) 


	# if using a GPU, assign seed and send model parameters to cuda
	if cuda:
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		model.cuda()


	# testing the model
	input_test = torch.randn([20, 3, 100, 100])
	output = model(input_test)



	return output.shape







################################################################################################

# calling the main function, this is where we assign variables

if __name__ == "__main__":

	output_shape = main(seed=0, cuda=False, C_in=3, nf1=6, k1=5, mpk1=2, mps1=2,
				nf2=9, k2=5, nf3=18, k3=5, fc1=256, fc2=128, fco=95)








