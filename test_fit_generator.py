import os
os.system('clear')
'''
    test fit generator
'''
#time and memory processing
#import time    
#    import psutil
#    py = psutil.Process(os.getpid())
#    memoryUse0 = py.memory_info()[0]/2.**30  # memory use in GB...I think
#    print('memory use:', memoryUse0)
    
    #function _srart
import numpy as np
from train_test_random_list import make_rand_list
from tpatch_batch_generator import train_patches_for_batches

addrs = '/home/seirana/Documents/Test_fit_generator/'
    
#for time_check in range(0,20):
#    time_start = time.clock()
    
patch_size = 256
batch_sz = 100
resolution = 0
array_length = 268
train_per = 0.8
make_rand_list(addrs, array_length, train_per)
patches = train_patches_for_batches(addrs, patch_size, batch_sz, resolution)

#    print('shape', np.shape(patches))
np.save(addrs+'patches.npy', patches)
    #function _end
    
    #time and memory processing
    #print('time_whole', time.clock() - time_start)
#    py = psutil.Process(os.getpid())
#    memoryUse1 = py.memory_info()[0]/2.**30  # memory use in GB...I think
#    print('memory use:', memoryUse1)