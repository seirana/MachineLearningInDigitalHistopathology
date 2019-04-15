"""
    fit generator
"""

# time and memory processing
# import time
#    import psutil
#    py = psutil.Process(os.getpid())
#    memoryUse0 = py.memory_info()[0]/2.**30  # memory use in GB...I think
#    print('memory use:', memoryUse0)
    
# function _srart
import numpy as np
import os
from patch_batch_generator import patches_for_batches
import h5py


def patch_generator(addrs, patch_size, batch_sz, resolution, tt):
    print('generator initiated')
    idx = 0
    while True:
        patches = patches_for_batches(addrs, patch_size, batch_sz, resolution, tt)  # produce patches for train/test
        patches = np.asarray(patches)
        print(np.shape(patches), tt)
        # if os.path.exists(addrs+'data.h5'):
        #     os.remove(addrs+'data.h5')
        # hf = h5py.File(addrs+'data.h5', 'w')
        # hf.create_dataset('dataset', data=patches)
        # hf.close()
        # with h5py.File(addrs+'data.h5', 'r') as hf:
        #     patches = hf.get('dataset')

        yield patches, patches
        print('generator yielded a batch %d' % idx, tt, np.shape(patches))
        idx += 1
