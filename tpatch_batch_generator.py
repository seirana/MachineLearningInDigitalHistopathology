## Importing the modules
import os
os.system('clear')

import numpy as np
import random
from OpenSlide_reader import read_openslide

'''
inputs:
patch_num: number of patches from each image we want to select for train and validation or test
img_lst: file with information of images: {number of objects, objects positions, threshhold}
result_addrs: the address to read convexhulls and ...
level_from: the level to extract small and grayscale patches
level_to: to conver the biggest image
'''

def train_patches_for_batches(addrs, img_lst, img_size, patch_size, rand_list, batch_sz, inChannel, all_patchs_per_image, train_per):
    #Create empty arrays to contain batch of features and labels
    batch_data = np.zeros((batch_sz, patch_size, patch_size, inChannel))
    for i in range(0,len(img_lst)):
        if rand_list[i] == 1:
            convex_hull = np.load(addrs + img_lst[i][0]) #read the convex_hull for the i-th image to check the convex hull coverage
            for b in range(0, img_lst[i][6]): 
                objct = read_openslide(img_lst[i][0], img_lst[i][7], b, addrs, addrs)
                extracted_patches = 0
                while extracted_patches < int(all_patchs_per_image/img_lst[i][6]):
                    x = random.randint(img_lst[i][7][0],img_lst[i][7][1]-patch_size) #produce a random number inside the object's rectangle, width
                    y = random.randint(img_lst[i][7][2],img_lst[i][7][3]-patch_size) #produce a random number inside the object's rectangle, height
                    c_patch = convex_hull[x:x+patch_size, y:y+patch_size]
                    patch_summation = c_patch.sum
                    #check_covexhull_coverage
                    if patch_summation == (patch_size^2)* 255: #the whole matrix must be inside the convex hull
                        patch = objct[x:x+patch_size,y:y+patch_size] #return a patch in level_to image 
                        batch_data.append(patch)
                        extracted_patches +=1

    return batch_data