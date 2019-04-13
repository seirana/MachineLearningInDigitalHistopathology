## Importing the modules
import os
os.system('clear')

import numpy as np
import random
from OpenSlide_reader import read_openslide
from Random_Rotation_mirroring import random_rot_mirr
from sizebased_sampling import make_random_list

'''
inputs:    
    result_addrs: the address to read information and files
    patch_size:size for patches 
    batch_sz: number of patches is needed for each batch
'''

def train_patches_for_batches(addrs, patch_size, batch_sz):
    '''slide_lst: file with information of slides:
       {slide name, number of resolusion levels, list of dimension resolusions, 
        resolusion us used for preprocessing, threshold, list of tissue pieces, 
        position of tissue piece}
    '''
    slide_lst= np.load(addrs+"image_list") #load a file to read information related to slides and tissue pieces on them
    train_test = np.load(addrs+"train_test_list") #read a list to know if the slide will be used for train or test
    slide_weight = make_random_list(addrs, batch_size) #
    batch_data = np.zeros((batch_sz, patch_size, patch_size)) #make a list to save patches for each batch
    
      ?????????????????????????????????????????????????????????????????????????  
            
            convex_hull = np.load(addrs + slide_lst[i][0]) #read the convex_hull for the i-th image to check the convex hull coverage
            for b in range(0, slide_lst[i][6]): 
                objct = read_openslide(slide_lst[i][0], slide_lst[i][7], b, addrs, addrs)
                extracted_patches = 0
                while extracted_patches < int(all_patchs_per_image/slide_lst[i][6]):
                    x = random.randint(slide_lst[i][7][0],slide_lst[i][7][1]-patch_size) #produce a random number inside the object's rectangle, width
                    y = random.randint(slide_lst[i][7][2],slide_lst[i][7][3]-patch_size) #produce a random number inside the object's rectangle, height
                    c_patch = convex_hull[x:x+patch_size, y:y+patch_size]
                    patch_summation = c_patch.sum
                    #check_covexhull_coverage
                    if patch_summation == (patch_size^2)* 255: #the whole matrix must be inside the convex hull
                        patch = objct[x:x+patch_size,y:y+patch_size] #return a patch in level_to image 
                        patch = random_rot_mirr(patch)
                        batch_data.append(patch)
                        extracted_patches +=1
                        

    return batch_data