"""
    fit generator
"""
import numpy as np
from patch_batch_generator import patches_for_batches


def patch_generator(addrs, patch_size, batch_sz, resolution, tt):
    # print('generator initiated')
    idx = 0
    while True:
        patches = patches_for_batches(addrs, patch_size, batch_sz, resolution, tt)  # produce patches for train/test
        patches = np.asarray(patches)
        # print(np.shape(patches), tt)
        yield patches, patches
        # print('generator yielded a batch %d' % idx)
        idx += 1
