'''
    Extract patches from slides
'''
import numpy as np
import random
from OpenSlide_reader import read_openslide

def get_patch_tissuepeices(addrs, slide_th, th_slide_info, slide_patch_num, batch_data,base_level, resolution, patch_size):
    tissues_slides = np.load(addrs+'tissues_slides_space_percentage.npy') #is a list

    whole_space = 0
    for i in range(0,len(tissues_slides[slide_th])):
        whole_space += tissues_slides[slide_th][i]

    sm = 0
    patch_tissue = np.zeros(len(tissues_slides[slide_th])) #number of patches we will extract of each tissue, based on tissue space size
    for i in range(0,len(tissues_slides[slide_th])-1):
        patch_tissue[i] = np.floor(slide_patch_num * tissues_slides[slide_th][i] / whole_space)
        sm += patch_tissue[i]
        
    patch_tissue[len(tissues_slides[slide_th])-1] = slide_patch_num - sm

    mul = 2^(base_level-resolution)
    
    margins = np.load(addrs + th_slide_info['slide_ID']+"_margin.npy") #read the margin for the i-th sldie to check the convex hull coverage    
    for tissue_th in margins.item(): #for all tissues in the slide   
        
        #multiply be 2^(base_level-resolution-base_level)
        up = th_slide_info['tissues'][tissue_th]['up']*mul
        down = (th_slide_info['tissues'][tissue_th]['down']+1)*mul-1
        left = th_slide_info['tissues'][tissue_th]['left']*mul
        right = (th_slide_info['tissues'][tissue_th]['right']+1)*mul-1 
        
        generated_patches = 0
        while generated_patches < patch_tissue[tissue_th]:
            #make random numbers inside the rectangle that coverd the tissue
            r_h=random.randint(up, down-patch_size+1)
            r_w=random.randint(left, right-patch_size+1)

            #check if the random numbers are inside the convex hull
            if np.floor(r_w/mul) >= margins.item().get(tissue_th)[0][0] and np.floor(r_w/mul) <= margins.item().get(tissue_th)[0][1]:
                if np.floor(r_h/mul) >= margins.item().get(tissue_th)[int(np.floor(r_w/mul))-margins.item().get(tissue_th)[0][0]][0] \
               and np.floor(r_h/mul) <= margins.item().get(tissue_th)[int(np.floor(r_w/mul))-margins.item().get(tissue_th)[0][0]][1]:
                    patch = read_openslide(addrs, th_slide_info['slide_ID'], up+r_h, left+r_w, patch_size) #extract patches
                    generated_patches += 1
                    batch_data.append(patch)

    return batch_data