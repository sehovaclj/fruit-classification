# Run this code second! 

# Author: Ljubisa Sehovac
# github: sehovaclj

# code that generates samples (classes and labels as well)  used to train, validate, and test the CNN model



# importing
import numpy as np

import torch

from PIL import Image

import glob
import os

import pickle


# change seed here to change order of random order of samples
# if you intend to have the model in fruit_class2.py see a different order of samples, change the seed here, run this script, then run fruit_class2.py 
seed=1

# seed == given seed above
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



# new datasets
images_final_test = images_testing[::10]
images_validation = images_testing.copy()
del images_validation[::10]




###########################################################################################

print("\nCreating train, validation, and test input samples + labels")


# generating training, validation, and testing input samples as well as labels
def generate_samples(dataset, normalize):

	# first make a copy of the dataset
	# d = dataset.copy()

	samples = []
	labels = []

	# select samples at random
	idxs = np.random.choice(len(dataset), len(dataset)) 

	# store the inputs and labels in lists
	for i in idxs:
		for key in dataset[i]:
			labels.append(key)
			if normalize:
				samples.append(dataset[i][key]/255.0) # might not need to normalize, pass True or False to function for normalize parameter
			elif not normalize:
				samples.append(dataset[i][key]) # normalizing to range [0, 1] by dividing by 255. Could also normalize using mean & std, but this should suffice.

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
class_labels_train, labels_onehot_train, samples_train = generate_samples(images_training, normalize=True)
class_labels_valid, labels_onehot_valid, samples_valid = generate_samples(images_validation, normalize=True)
class_labels_test, labels_onehot_test, samples_test = generate_samples(images_final_test, normalize=True)





####################################################################################

# pickling


with open('class_labels_train.pkl', 'wb') as f:
	pickle.dump(class_labels_train, f)
with open('labels_onehot_train.pkl', 'wb') as f:
	pickle.dump(labels_onehot_train, f)
with open('samples_train.pkl', 'wb') as f:
	pickle.dump(samples_train, f, protocol=4)

with open('class_labels_valid.pkl', 'wb') as f:
	pickle.dump(class_labels_valid, f)
with open('labels_onehot_valid.pkl', 'wb') as f:
	pickle.dump(labels_onehot_valid, f)
with open('samples_valid.pkl', 'wb') as f:
	pickle.dump(samples_valid, f)

with open('class_labels_test.pkl', 'wb') as f:
	pickle.dump(class_labels_test, f)
with open('labels_onehot_test.pkl', 'wb') as f:
	pickle.dump(labels_onehot_test, f)
with open('samples_test.pkl', 'wb') as f:
	pickle.dump(samples_test, f)




# end of script

