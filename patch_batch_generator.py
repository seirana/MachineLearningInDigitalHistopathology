# Importing the modules
import numpy as np
from sizebased_sampling import make_random_list
from get_patch_tissues import get_patch_tissuepeices
"""
inputs:    
    result_addrs: the address to read information and files
    patch_size:size for patches 
    batch_sz: number of patches is needed for each batch
"""


def patches_for_batches(addrs, patch_size, batch_sz, numof_imgs, resolution, tt, batch_num):
    """
    slides: file with information of slides:
       {slide name, number of resolusion levels, list of dimension resolusions, 
        resolusion us used for preprocessing, threshold, list of tissue pieces, 
        position of tissue piece}
    """
    # load a file to read information related to slides and tissue pieces on them
    slides = np.load(addrs+'_SlidesInfo_dic.npy')
    # read a list to know if the slide will be used for train or test
    train_test = np.load(addrs+'train_test_list.npy')[()]
    # make a random list, to split the batch size for extracting random number of patches from slides

    slide_th = make_random_list(addrs, batch_sz, tt)

    batch_data = list()  # make a list to save patches for each batch
    if tt == 'train':
        if train_test[slide_th] == 'train':  # if the slide was selected for train
            batch_data = get_patch_tissuepeices(addrs, slide_th, slides.item().get(slide_th),
                                                batch_sz,
                                                batch_data,
                                                slides.item().get(slide_th)['magnification_level'],
                                                # extract patches from tissues from all slides
                                                resolution, 
                                                patch_size, 
                                                tt)
    if tt == 'test':
        if train_test[slide_th] == 'test':  # if the slide was selected for test
            batch_data = get_patch_tissuepeices(addrs, slide_th, slides.item().get(slide_th),
                                                batch_sz,
                                                batch_data,
                                                slides.item().get(slide_th)['magnification_level'],
                                                # extract patches from tissues from all slides
                                                resolution, 
                                                patch_size, 
                                                tt)

    return batch_data
