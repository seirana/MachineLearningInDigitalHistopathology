""" Build and train a convolutional neural network with TensorFlow."""
import os
import openslide
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
from PIL import Image
import Random_Rotation_Mirroring as RRM ##mine

batch_size = tf.constant(1)        
ttt_ratio = tf.constant(0.33) #train to test ration


#serching for special files to read 
src_addrs = "/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/Results/"
file_list = np.load(src_addrs+"_info_gathering.npy")


num_of_sample_file = 0
Casp3_list = tf.placeholder(tf.int32, [None])
if file.sratrswith(src_addrs + "patch_list") and file.endswith(".npy"): 
    for i in range(0,len(file_list)):
        if "Casp3" in file_list[i][0]:            
            Casp3_list[num_of_sample_file] = i
            num_of_sample_file += 1


patch_size = tf.constant(file_list[0][9])
feasure_size = tf.constant(patch_size**2)
epochs = tf.constant(feasure_size)
learning_rate = tf.constant(1/(feasure_size))


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=ttt_ratio)

rand_sel_train_files = tf.random_uniform(np.empty[0,math.ceil(num_of_sample_file*ttt_ratio)], minval=0, maxval=num_of_sample_file, dtype=tf.int32)
sel_patches_each_batch = tf.contrib.framework.sort(rand_sel_train_files,axis=0,direction='ASCENDING')

            
for t in range(0,k):
    if i < cnt+len(file):
        file = result_addrs + "patch_list" + str(k) + ".ndpi"
        loaded_mat = np.load(file)
        tmp = loaded_mat[i-cnt,:,:,:]
        train_set.append(RRM.random_rot_mirr(tmp[i-cnt,:,:,:]))

"""It id done!"""