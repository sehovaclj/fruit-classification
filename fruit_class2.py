# run this code third!

# Author: Ljubisa Sehovac
# github: sehovaclj
# email: lsehovac@uwo.ca

# feel free to email me anytime regarding my code, any questions you may have, how I went about doing things, etc.


# python code that: 
#	contains CNN model to be used for fruit classification
#	training, validation, and update of parameters


# Note: a grid search has not yet been implemented (hyperparameter optimization to achieve the best model)

# importing functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import init

import random
import time

import math

from PIL import Image

import glob
import os

import pickle

import sys


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

def main(seed, cuda, arc, loss_function_choice, epochs, batch_size, 
		learning_rate, opt, mtum, rdction, test_four):


	# seed == given seed
	np.random.seed(seed)
	torch.manual_seed(seed)


	##########################################################################################
	# first thing we have to do is load the training and testing images from the pickle files

	print("\nLoading pickle files")

	# load pickle files
	with open('class_labels_train.pkl', 'rb') as f:
		class_labels_train = pickle.load(f)
	with open('labels_onehot_train.pkl', 'rb') as f:
		labels_onehot_train = pickle.load(f)
	with open('samples_train.pkl', 'rb') as f:
		samples_train = pickle.load(f)

	with open('class_labels_valid.pkl', 'rb') as f:
		class_labels_valid = pickle.load(f)
	with open('labels_onehot_valid.pkl', 'rb') as f:
		labels_onehot_valid = pickle.load(f)
	with open('samples_valid.pkl', 'rb') as f:
		samples_valid = pickle.load(f)
	
	with open('class_labels_test.pkl', 'rb') as f:
		class_labels_test = pickle.load(f)
	with open('labels_onehot_test.pkl', 'rb') as f:
		labels_onehot_test = pickle.load(f)
	with open('samples_test.pkl', 'rb') as f:
		samples_test = pickle.load(f)
	

	with open('classes.pkl', 'rb') as f:
		classes = pickle.load(f)



	############################################################################################

	print("\nCalling the model")
	
	# calling the model
	model = CNN_Net(
		arc['C_in'],
		arc['num_filters1'],
		arc['k1'],
		arc['maxpool_k1'],
		arc['maxpool_s1'],
		arc['num_filters2'],
		arc['k2'],
		arc['num_filters3'],
		arc['k3'],
		arc['fc_size1'],
		arc['fc_size2'],
		arc['fc_output']
		) 

	# if using a GPU, assign seed and send model parameters to cuda
	if cuda:
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		model.cuda()

	print("\n")
	print(model)

	# testing the model
	#input_test = torch.randn([20, 3, 100, 100])
	#output = model(input_test)


	# if statement choosing the optimizer and loss function
	if opt == 'Adam':
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
	if opt == 'SGD':
		optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=mtum)
	
	# if statements to choose loss function: MSE(when using one-hot as target) or CE (when using class labels as target)
	# it is left as an option for the reader to experiment with different loss functions, or implement different ones themselves using similar syntax
	if loss_function_choice == 'MSE':
 		loss_fn = nn.MSELoss(reduction=rdction) 
	if loss_function_choice == 'CE':
		loss_fn = nn.CrossEntropyLoss(reduction=rdction)


	print("\nStarting training")

	train_loss = []
	valid_loss = []


	# loop over each epoch
	for epoch in range(epochs):

		t_one_epoch = time.time()

		print("\nEpoch {}".format(epoch+1))

		train_epoch_loss = 0

		###############################################################################

		# TRAINING

		# loop over each batch
		for b_idx in range(0, len(samples_train), batch_size):

			# obtain input and target samples for each batch
			inputs = samples_train[b_idx:b_idx+batch_size]
			if loss_function_choice == 'MSE':
				targets = labels_onehot_train[b_idx:b_idx+batch_size]
			if loss_function_choice == 'CE':
				targets = class_labels_train[b_idx:b_idx+batch_size].view(-1) 

			# use cuda if available
			if cuda:
				inputs = inputs.cuda()
				targets = targets.cuda()

			# zero parameter gradients
			optimizer.zero_grad()

			# forward pass of batch inputs through CNN model to obtain outputs
			outputs = model(inputs)			

			# if MSE loss, convert outputs to onehot to compare with target

			# compute loss between outputs and targets
			loss = loss_fn(outputs, targets)
			
			# save loss of this batch and add it to total loss of epoch
			train_epoch_loss += loss.item()

			# backward + optimize
			loss.backward()
			optimizer.step()


		# append total loss of epoch
		train_loss.append(train_epoch_loss)

		print("TRAINING LOSS: {}".format(train_epoch_loss))
			
		####################################################################################

		# TESTING -- essentially same as above, but don't backward+optimize

		valid_epoch_loss = 0

		total_correct = 0
		total_samples = 0

		for b_idx in range(0, len(samples_valid), batch_size):
			with torch.no_grad():

				inputs = samples_valid[b_idx:b_idx+batch_size]
				if loss_function_choice == 'MSE':
					targets = labels_onehot_valid[b_idx:b_idx+batch_size]
				if loss_function_choice == 'CE':
					targets = class_labels_valid[b_idx:b_idx+batch_size].view(-1) 

				if cuda:
					inputs = inputs.cuda()
					targets = targets.cuda()

				outputs = model(inputs)

				loss = loss_fn(outputs, targets)

				valid_epoch_loss += loss.item()

				# calculating accuracy
				predicted_class = outputs.argmax(dim=1).cpu()
				actual_class = class_labels_valid[b_idx:b_idx+batch_size].view(-1)

				correct = (predicted_class == actual_class).sum()
				out_of = batch_size # don't really need this, but good practice

				total_correct += correct.item()
				total_samples += out_of

		# append total loss of test 
		valid_loss.append(valid_epoch_loss)

		# All train and validation batches passed, one epoch complete here
		# print statements here
		print("TESTING LOSS: {}".format(valid_epoch_loss))
		print("\nAccuracy for validation set: {} correct / {} total samples = {}%".format( total_correct, total_samples, (total_correct/total_samples)*100.0 ))



	############################################################################################################

	# Test the model on first four samples from test set (remember, samples are randomized)
	# This test is done after model has completed training

	if test_four:

		with torch.no_grad():

			inputs_test = samples_test[:4]
			actual_class_test = class_labels_test[:4]

			if cuda:
				inputs_test = inputs_test.cuda()

			outputs_test = model(inputs_test)

			predicted_class_test = torch.zeros(outputs_test.shape[0]).view(-1, 1).to(torch.long) # empty zero tensor the same length as outputs 	
			for i in range(len(outputs_test)):
				predicted_class_test[i] = outputs_test[i].argmax()

			# this is where we connect back to original String classes, see loading pickle files near top
			print("\nModel has predicted the classes as:\t{},\t{},\t{},\t{}".format( classes[predicted_class_test[0]],
				classes[predicted_class_test[1]], classes[predicted_class_test[2]], classes[predicted_class_test[3]] ))
			print("\nActual classes are:                \t{},\t{},\t{},\t{}".format( classes[actual_class_test[0]],
				classes[actual_class_test[1]], classes[actual_class_test[2]], classes[actual_class_test[3]] ))




	# end of main function
	# return desired variables

	return model, train_loss, valid_loss, total_correct, total_samples 







################################################################################################

# assigning the model architecture

MODEL_ARC = {
	'C_in': 3,
	'num_filters1': 9,
	'k1': 5,
	'maxpool_k1': 2,
	'maxpool_s1': 2,
	'num_filters2': 18,
	'k2': 5,
	'num_filters3': 36,
	'k3': 5,	
	'fc_size1': 512,
	'fc_size2': 256,
	'fc_output': 95
	}
	 

# calling the main function, this is where we assign variables
# shouldn't train for more than 20 epochs, 10 is usually sufficient to achieve model that does not overfit and generalizes well 
if __name__ == "__main__":

	MODEL, train_loss, valid_loss, correct, sampled = main(seed=1, cuda=True,
		arc=MODEL_ARC, loss_function_choice='MSE', epochs=20,
		batch_size=256, learning_rate=0.0005, opt='Adam', mtum=0.9,
		rdction='sum', test_four=True)







# end of script
