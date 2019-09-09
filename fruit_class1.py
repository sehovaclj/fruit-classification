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

def main(seed, cuda, arc, loss_function_choice, epochs, batch_size, test_four):


	# seed == given seed
	np.random.seed(seed)
	torch.manual_seed(seed)



	##########################################################################################
	# first thing we have to do is load the training and testing images from the pickle files

	print("\nLoading pickle files")

	# load pickle files
	with open('images_training.pkl', 'rb') as f:
		images_training = pickle.load(f)

	with open('images_testing.pkl', 'rb') as f:
		images_testing = pickle.load(f)

	with open('classes.pkl', 'rb') as f:
		classes = pickle.load(f)


	# creating a validation and test set. Primarily be testing on validation set, leave test set for very end
	# every 10th element we will store as the test set. Hence 16421 total in test set, now 1643 in test set and 14778 in validation set.
	images_final_test = images_testing[::10]
	images_validation = images_testing.copy()
	del images_validation[::10]


	###########################################################################################

	print("\nCreating train, validation, and test input samples + labels")


	# generating training, validation, and testing input samples as well as labels
	def generate_samples(dataset):

		# first make a copy of the dataset
		d = dataset.copy()

		samples = []
		labels = []

		# select samples at random
		idxs = np.random.choice(len(d), len(d)) 

		# store the inputs and labels in lists
		for i in idxs:
			for key in d[i]:
				labels.append(key)
				samples.append(d[i][key])

		labels = torch.Tensor(labels).view(-1, 1).to(torch.long) # need this for one-hot encoding

		N = len(labels) # total length of samples
		num_classes = 95 # total number of classes

		# convert each class label to one-hot encoding label
		labels_onehot = torch.FloatTensor(N, num_classes).zero_() # matrix of dimension N x 95
		labels_onehot.scatter_(1, labels, 1) # one-hot here


		# dimensions of image should be 3x100x100
		depth = samples[0].shape[0]
		h = samples[0].shape[1]
		w = samples[0].shape[2]

		# empty tensor matrix to store tensors from samples list
		S = torch.zeros([len(samples), depth, h, w])
		# convert list of tensors (samples) to tensor
		for i in range(len(samples)):
			S[i] = samples[i]


		return labels, labels_onehot, S



	# call function three times for three different datasets
	class_labels_train, labels_onehot_train, samples_train = generate_samples(images_training)
	class_labels_valid, labels_onehot_valid, samples_valid = generate_samples(images_validation)
	class_labels_test, labels_onehot_test, samples_test = generate_samples(images_final_test)


 

	############################################################################################
	
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


	# testing the model
	#input_test = torch.randn([20, 3, 100, 100])
	#output = model(input_test)


	# stating the optimizer and loss function
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	# if statements to choose loss function, either MSE(when using one-hot as target) or CE (when using class labels as target)
	if loss_function_choice == 'MSE':
 		loss_fn = nn.MSELoss() 
	if loss_function_choice == 'CE':
		loss_fn = nn.CrossEntropyLoss()


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
				targets = class_labels_train[b_idx:b_idx+batch_size] 

			# use cuda if available
			if cuda:
				inputs = inputs.cuda()
				targets = targets.cuda()

			# zero parameter gradients
			optimizer.zero_grad()

			# forward pass of batch inputs through CNN model to obtain outputs
			outputs = model(inputs)			

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
					targets = class_labels_valid[b_idx:b_idx+batch_size] 

				if cuda:
					inputs = inputs.cuda()
					targets = targets.cuda()

				outputs = model(inputs)

				loss = loss_fn(outputs, targets)

				valid_epoch_loss += loss.item()

				# calculating accuracy
				predicted_class = torch.zeros(outputs.shape[0]).view(-1, 1).to(torch.long) # empty zero tensor the same length as outputs 	
				for i in range(len(outputs)):
					predicted_class[i] = outputs[i].argmax()

				actual_class = class_labels_valid[b_idx:b_idx+batch_size]

				correct = sum(predicted_class == actual_class)
				out_of = batch_size # don't really need this, but good practice

				total_correct += correct.item()
				total_samples += out_of.item()	# don't really need this, but good practice

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









	return model, class_labels_train, labels_onehot_train, samples_train, total_correct, total_samples 







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

if __name__ == "__main__":

	MODEL, classes, labels, samples, correct, sampled = main(seed=1, cuda=True,
		arc=MODEL_ARC, loss_function_choice='MSE', epochs=20,
		batch_size=64, test_four=True)







