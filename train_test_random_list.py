'''
This function makes a random list of 0s or 1s in the size of number of slides we have in total for train and test. 
if it is 1 then the slide will go to train set else for the test set.
e.g if train_tes[i] == 1 then i-th slide will use as train data
'''
## Importing the modules
import numpy as np
#from random import randint

def make_rand_list(addrs, array_length, train_per):
    
    train_test = np.zeros(shape=(array_length))
    train_test[1] = 1
    train_test[25] = 1   
    np.save(addrs+'train_test_list.npy', train_test)   
    
#    train_test = np.zeros(shape=(array_length))
#    c = 0
#    while c < train_per*array_length: #train_per = the percentage of openslise slides we will use for train, e.g 80%
#        r = randint(0,array_length-1)
#        if train_test[r] == 0:
#            train_test[r] = 1
#            c+=1
#    
#    dictOftrain_tests = {i:train_test[i] for i in range(0,len(train_test))}       
#    np.save(addrs+"train_test_list", dictOftrain_tests)       
#    return 
#

##checked works well