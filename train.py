
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import os
import numpy as np 
import itertools
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from keras.models import Sequential 
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense 
from keras import applications 
from keras.utils.np_utils import to_categorical 

# Part 1
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


# Get the number of classes

num_classes = 19

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = num_classes, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam',loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 1 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(
    'dataset_images/training_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'categorical')

# get the class labels for the training data, in the original order 
train_labels = training_set.classes 
 
# convert the training labels to categorical vectors 
train_labels = to_categorical(train_labels, num_classes=num_classes)
#print (train_labels)

test_set = test_datagen.flow_from_directory(
    'dataset_images/test_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'categorical')


classifier.fit_generator( training_set,
    steps_per_epoch = 8000,
    epochs = 25,
    validation_data = test_set,
    validation_steps = 2000)


import joblib

joblib.dump(classifier, 'cnn_classifier')