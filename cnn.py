#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 12:51:08 2019

"""

#Import Libraries
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
import os
import numpy as np 
import itertools
import keras
import tensorflow.keras.backend as K
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense 
from keras import applications 
from keras.utils.np_utils import to_categorical 

# Part 1 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

test_datagen = ImageDataGenerator()

training_set = train_datagen.flow_from_directory(
    'dataset_images/training_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'categorical')

# get the class labels for the training data, in the original order 
train_labels = training_set.classes 
print (train_labels)

nb_train_samples = len(training_set.filenames) 
num_classes = len(training_set.class_indices) 

test_set = test_datagen.flow_from_directory(
    'dataset_images/test_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'categorical')

nb_test_samples = len(test_set.filenames)
num_classes_test = len(test_set.class_indices)

print('Training Samples: {} and Classes: {}'.format(nb_train_samples, num_classes))
print('Test Samples: {} and Classes: {}'.format(nb_test_samples, num_classes_test))


# Part 2
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = num_classes, activation = 'softmax'))

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Compiling the CNN
classifier.compile(optimizer=sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])


classifier.fit_generator( training_set,
    steps_per_epoch = 19118,
    epochs = 25,
    workers = 6,
    validation_data = test_set,
    validation_steps = 4791)


import joblib

model = 'cnn_classifier.jb'

joblib.dump(classifier, model)


import pickle

with open('cnn_classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)