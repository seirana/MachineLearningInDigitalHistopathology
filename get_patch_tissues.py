'''
    Extract patches from slides
'''
import numpy as np
import random
from OpenSlide_reader import read_openslide

def get_patch_tissuepeices(addrs, th_slide_info, slide_patch_size, batch_data,resolution, patch_size):
        
    margins = np.load(addrs + th_slide_info['slide_ID']+"_margin.npy") #read the margin for the i-th sldie to check the convex hull coverage
    tissue_count = margins.item().get('tissue_count')
    base_level = margins.item().get('magnification_level')
    mul = 2^(base_level-resolution-base_level)
    
    for tissue_th in margins.item().get('tissues'):
        tissue_space = np.zeros(tissue_count)
        whole_space = 0
        for i in range(0, tissue_count): #make summation of the whole tissue space in the slide
            tissue_space[i] += ((margins.item().get('tissues')['down'] - margins.item().get('tissues')['up']+1) * (margins.item().get('tissues')['right'] - margins.item().get('tissues')['left']+1))        
            whole_space += tissue_space[i]
        
        sm = 0
        patch_tissue = list() #number of patches we will extract of each tissue, based on tissue space size
        for i in range(0,tissue_count-1):
            patch_tissue[i] = np.floor(slide_patch_size * tissue_space[i] / whole_space)
            sm += patch_tissue[i]
        patch_tissue[tissue_count-1] = slide_patch_size - sm
        
    for tissue_th in margins.item().get('tissues'): #for all tissues in the slide
        
        #multiply be 2^(base_level-resolution-base_level)
        up = margins.item().get('tissues')['up']*mul
        down = (margins.item().get('tissues')['down']+1)*mul-1
        left = margins.item().get('tissues')['left']*mul
        right = (margins.item().get('tissues')['right']+1)*mul-1
        
        generated_patches = 0
        while generated_patches < patch_tissue[i]:
            #make random numbers inside the rectangle that coverd the tissue
            r_w=random.randint(up, down-patch_size+1)
            r_h=random.randint(left, right-patch_size+1)
                    
            #check if the random numbers are inside the convex hull
            if np.floor(r_w/mul) >= margins.item().get('tissues')[0][0] and np.floor(r_w/mul) <= margins.item().get('tissues')[0][1]:
                if np.floor(r_h/mul) >= margins.item().get('tissues')[np.floor(r_w/mul)][0] and np.floor(r_h/mul) >= margins.item().get('tissues')[np.floor(r_w/mul)][1]:    
                    patch = read_openslide(addrs, th_slide_info['slide_ID'], r_w, r_h, patch_size) #extract patches
                    generated_patches += 1
                    batch_data.apend(patch)
                    
    return batch_data