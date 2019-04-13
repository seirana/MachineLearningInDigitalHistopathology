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
from patch_batch_generator import patches_for_batches

def patch_generator(addrs, patch_size, batch_sz, resolution, array_length, train_per, tt):
    
    if tt == 0:        
        patches = patches_for_batches(addrs, patch_size, batch_sz, resolution, 0) ##produce patches for train
    else:
        patches = patches_for_batches(addrs, patch_size, batch_sz, resolution, 1)##produce patches for test
            
    return patches
