'''
    Extract patches from slides
'''
import numpy as np
import random
from OpenSlide_reader import read_openslide

def get_patch_tissuepeices(addrs, th_slide_info, slide_patch_size, batch_data,resolution, patch_size):
        
    margins = np.load(addrs + th_slide_info['slide_ID']+"_margin") #read the margin for the i-th sldie to check the convex hull coverage
    tissue_count = th_slide_info['tissue_count']
    base_level = th_slide_info['magnification_level']
    mul = 2^(base_level-resolution-base_level)
    
    for tissue_th in th_slide_info['tissues']:
        tissue_space = np.zeros(tissue_count)
        whole_space = 0
        for i in range(0, tissue_count): #make summation of the whole tissue space in the slide
            tissue_space[i] += ((th_slide_info['tissues']['down'] - th_slide_info['tissues']['up']+1) * (th_slide_info['tissues']['right'] - th_slide_info['tissues']['left']+1))        
            whole_space += tissue_space[i]
        
        sm = 0
        patch_tissue = list() #number of patches we will extract of each tissue, based on tissue space size
        for i in range(0,tissue_count-1):
            patch_tissue[i] = np.floor(slide_patch_size * tissue_space[i] / whole_space)
            sm += patch_tissue[i]
        patch_tissue[tissue_count-1] = slide_patch_size - sm
        
    for tissue_th in th_slide_info['tissues']: #for all tissues in the slide
        
        #multiply be 2^(base_level-resolution-base_level)
        up = th_slide_info['tissues']['up']*mul
        down = (th_slide_info['tissues']['down']+1)*mul-1
        left = th_slide_info['tissues']['left']*mul
        right = (th_slide_info['tissues']['right']+1)*mul-1
        
        generated_patches = 0
        while generated_patches < patch_tissue[i]:
            #make random numbers inside the rectangle that coverd the tissue
            r_w=random.randint(up, down-patch_size+1)
            r_h=random.randint(left, right-patch_size+1)
                    
            #check if the random numbers are inside the convex hull
            if r_h >= margins[tissue_th][r_w+1][0] and r_h <= margins[tissue_th][r_w+1][1]:
                patch = read_openslide(addrs, th_slide_info['slide_ID'], r_w, r_h, patch_size) #extract patches
                generated_patches += 1
                batch_data.apend(patch)
                    
    return batch_data                