'''
This function makes a random list of 0s or 1s in the size of number of images we have in total for train or test. 
if it is 1 then the image will go to train set else for the test set.
e.g if train_tes[i] == 1 then i-th image will use as train data
'''
## Importing the modules
import os
os.system('clear')

import numpy as np
from random import randint

def make_rand_list(array_length, train_per):
    train_test = np.zeros(shape=(array_length))
    c = 0
    while c < train_per*array_length: #train_per = the percentage of openslise images we will use for train, e.g 80%
        r = randint(0,array_length)
        if train_test[r] == 0:
            train_test[r] == 1
            c+=1
            
    return train_test         