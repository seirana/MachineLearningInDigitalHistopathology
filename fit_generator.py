'''
    fit generator
'''
#time and memory processing
#import time    
#    import psutil
#    py = psutil.Process(os.getpid())
#    memoryUse0 = py.memory_info()[0]/2.**30  # memory use in GB...I think
#    print('memory use:', memoryUse0)
    
#function _srart
import numpy as np
from patch_batch_generator import patches_for_batches

def patch_generator(addrs, patch_size, batch_sz, resolution, tt):
    print('generator initiated')
    idx = 0
    while True:
        patches = patches_for_batches(addrs, patch_size, batch_sz, resolution, tt) ##produce patches for train/test        
        print(tt)
        patches = np.asarray(patches)
        yield patches, patches
        print("patch",type(patches), np.shape(patches))
        print('generator yielded a batch %d' % idx)
        idx += 1