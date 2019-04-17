"""
    Extract patches from slides
"""
import numpy as np
import random
from OpenSlide_reader import read_openslide


def get_patch_tissuepeices(addrs, slide_th, th_slide_info, slide_patch_num, 
                           batch_data, base_level, resolution, patch_size, tt):
    patch_list = np.load(addrs+'patch_train_test_list.npy')
    tissues_slides = np.load(addrs+'tissues_slides_space_percentage.npy')  # is a list
    whole_space = 0
    for i in range(0, len(tissues_slides[slide_th])):
        whole_space += tissues_slides[slide_th][i]

    sm = 0
    # number of patches we will extract of each tissue, based on tissue space size
    patch_tissue = np.zeros(len(tissues_slides[slide_th]))
    for i in range(0, len(tissues_slides[slide_th])-1):
        patch_tissue[i] = np.floor(slide_patch_num * tissues_slides[slide_th][i] / whole_space)
        sm += patch_tissue[i]
        
    patch_tissue[len(tissues_slides[slide_th])-1] = slide_patch_num - sm

    mul = 2**(base_level-resolution)
    # read the margin for the i-th sldie to check the convex hull coverage
    margins = np.load(addrs + th_slide_info['slide_ID']+"_margin.npy")
    for tissue_th in margins.item():  # for all tissues in the slide
        
        up = th_slide_info['tissues'][tissue_th]['up']
        down = th_slide_info['tissues'][tissue_th]['down']
        left = th_slide_info['tissues'][tissue_th]['left']
        right = th_slide_info['tissues'][tissue_th]['right']
        
        generated_patches = 0
        while generated_patches < patch_tissue[tissue_th]:
            # make random numbers inside the rectangle that coverd the tissue
            r_h = random.randint(up, down)
            r_w = random.randint(left, right)

            # check if the random numbers are inside the convex hull
            if margins.item().get(tissue_th)[0][0] <= r_w <= margins.item().get(tissue_th)[0][1]:
                if margins.item().get(tissue_th)[r_w+1-margins.item().get(tissue_th)[0][0]][0] <= r_h \
                        <= margins.item().get(tissue_th)[r_w+1-margins.item().get(tissue_th)[0][0]][1]: 
                            
                    #this is to be sure that the randomly selected patches do not have overlap    
                    while True:      
                        r_h_mul = random.randint(0, mul-1)
                        for patch in patch_list.item():
                            if slide_th == patch['slide_number']:
                                if -256 < r_h*mul+r_h_mul - patch['height'] < 256:
                                    r_w_mul = random.randint(0, mul-1)
                                    if -256 < r_w*mul+r_w_mul - patch['weidth'] < 256:    
                                        break
        
                    # multiply by 2**(base_level-resolution-base_level)
                    patch = read_openslide(addrs, 
                                           slide_th, 
                                           th_slide_info['slide_ID'], 
                                           r_h*mul+r_h_mul, 
                                           r_w*mul+r_w_mul,
                                           patch_size, 
                                           tt)  # extract patches
                    generated_patches += 1
                    batch_data.append(patch)
                    
    return batch_data
