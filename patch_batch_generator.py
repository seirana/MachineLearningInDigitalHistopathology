## Importing the modules
import numpy as np
from sizebased_sampling import make_random_list
from get_patch_tissues import get_patch_tissuepeices
'''
inputs:    
    result_addrs: the address to read information and files
    patch_size:size for patches 
    batch_sz: number of patches is needed for each batch
'''

def patches_for_batches(addrs, patch_size, batch_sz, resolution, tt):
    '''slides: file with information of slides:
       {slide name, number of resolusion levels, list of dimension resolusions, 
        resolusion us used for preprocessing, threshold, list of tissue pieces, 
        position of tissue piece}
    '''
    slides= np.load(addrs+'_SlidesInfo_dic.npy') #load a file to read information related to slides and tissue pieces on them
    train_test = np.load(addrs+'train_test_list.npy') #read a list to know if the slide will be used for train or test
    slide_weight = make_random_list(addrs, batch_sz) #make a random list, to split the batch size for extracting random number of patches from slides
    batch_data = list() #make a list to save patches for each batch
    for slide_th in slides.item(): #for all silde
        if tt == 'train':
            print(slide_th)
            if train_test[slide_th] == 'train': #if the slide was selected for train
                batch_data = get_patch_tissuepeices(addrs, slide_th, slides.item().get(slide_th),\
                                                    slide_weight[slide_th],\
                                                    batch_data,\
                                                    slides.item().get(slide_th)['magnification_level'],\
                                                    resolution, patch_size) #extract patches from tissues from all slides
        else:
            if train_test[slide_th] == 'test': #if the slide was selected for test
                batch_data = get_patch_tissuepeices(addrs, slide_th, slides.item().get(slide_th),\
                                                    slide_weight[slide_th],\
                                                    batch_data,\
                                                    slides.item().get(slide_th)['magnification_level'],\
                                                    resolution, patch_size) #extract patches from tissues from all slides

    return batch_data