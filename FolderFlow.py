from keras.preprocessing.image import ImageDataGenerator
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import numpy as np
import resnet
import os
import cv2
import csv

img_addrs = "/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/Test/"
batch_size = 128
shp = np.shape(images)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(\
        rescale=1./255,\
        shear_range=0,\
        zoom_range=0,\
        horizontal_flip=False,\
        width_shift_range=0.1, height_shift_range=0.1) # randomly shift images horizontally/vertically (fraction of total width/height)               

# generator for reading train data from folder
"""
 this is a generator that will read pictures found in
 subfolers of '/train', and indefinitely generate
 batches of augmented image data
 """
train_generator = train_datagen.flow_from_directory(\
        directory= img_addrs + "train/",\
        rescale=1./255,\
        shear_range=0.2,\
        zoom_range=0.2,\
        horizontal_flip=True,\
        target_size=(shp[1], shp[2]),\
        color_mode="rgb",\
        batch_size=batch_size,\
        class_mode="input",\
        shuffle=True,\
        seed=42)

# this is the augmentation configuration we will use for validation:
# only rescaling        
valid_datagen = ImageDataGenerator(rescale=1./255)
 
# generator for reading validation data from folder
valid_generator = valid_datagen.flow_from_directory(\
    directory= img_addrs + "valid/",\
    target_size=(shp[1], shp[2]),\
    color_mode="rgb",\
    batch_size=batch_size,\
    class_mode="input",\
    shuffle=True,\
    seed=42)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# generator for reading test data from folder
# this is a similar generator, for validation data
test_generator = test_datagen.flow_from_directory(\
    directory= img_addrs + "test/",\
    target_size=(shp[1], shp[2]),\
    color_mode="rgb",\
    batch_size=1,\
    class_mode=None,\
    shuffle=False,\
    seed=42) 

autoencoder.fit_generator(\
        train_generator,
        steps_per_epoch=2000 // batch_size,\
        epochs=50,\
        validation_data=valid_generatorr,\
        validation_steps=800 // batch_size)
        
autoencoder.save_weights('first_try.h5')  # always save your weights after training or during training