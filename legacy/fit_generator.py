"""
    fit generator
"""
import numpy as np
from patch_batch_generator import patches_for_batches


def patch_generator(addrs, patch_size, batch_sz, numof_imgs, resolution, tt):
    # print('generator initiated')
    batch_num = 0
    while True:
        # produce patches for train/test
        patches = patches_for_batches(addrs, patch_size, batch_sz, numof_imgs, resolution, tt, batch_num)
        patches = np.asarray(patches)
        # print(np.shape(patches), tt)
        yield patches, patches
        # print('generator yielded a batch %d' % batch_num)
        batch_num += 1
