import os
os.system('clear')
'''
    test fit generator
'''
#import psutil
#print(psutil.virtual_memory())
#print(psutil.cpu_times())
#print(psutil.cpu_freq(percpu=True))
import time
time_start = time.clock()
import resource
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

import numpy as np
from train_test_random_list import make_rand_list
from tpatch_batch_generator import train_patches_for_batches

addrs = '/home/seirana/Documents/Test_fit_generator/'
  
patch_size = 256
batch_sz = 10000
resolution = 0
array_length = 268
train_per = 0.8
make_rand_list(addrs, array_length, train_per)
patches = train_patches_for_batches(addrs, patch_size, batch_sz, resolution)

#print(psutil.virtual_memory())
#print(psutil.cpu_times())
#print(psutil.cpu_freq(percpu=True))
time_elapsed = (time.clock() - time_start)
print('time', time_elapsed)
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

print('shape', np.shape(patches))
np.save(addrs+'patches.npy', patches)