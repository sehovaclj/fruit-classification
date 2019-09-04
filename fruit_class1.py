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





############################################################################################

# building the CNN model




























############################################################################################

# first thing we have to do is load the training and testing images from the pickle files

# load pickle files
with open('images_training.pkl', 'rb') as f:
	images_training = pickle.load(f)

with open('images_testing.pkl', 'rb') as f:
	images_testing = pickle.load(f)

with open('classes.pkl', 'rb') as f:
	classes = pickle.load(f)






