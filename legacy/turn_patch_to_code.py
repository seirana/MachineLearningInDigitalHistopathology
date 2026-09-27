import h5py
import numpy as np


def turn_patch_to_code():
    
    addrs = ""
    with h5py.File('data.h5','r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = hf.get('dataset_1')
        np_data = np.array(data)
        print('Shape of the array dataset_1: \n', np_data.shape)
    
    return
