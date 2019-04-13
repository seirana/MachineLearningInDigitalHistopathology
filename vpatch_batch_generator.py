## Importing the modules
import os
os.system('clear')

import numpy as np
import math
from random import randint
from numpy.matrix import sum
from OpenSlide_reader import read_openslide
'''
inputs:
patch_num: number of patches from each image we want to select for train and validation or test
information: file with information of images: {number of objects, objects positions, threshhold}
result_addrs: the address to read convexhulls and ...
level_from: the level to extract small and grayscale patches
level_to: to conver the biggest image
'''

def one_patch(patch_num, information, result_addrs, level_from, level_to, patch_size):
    mini_patch_size = 8 #patch_size / 2^(level_to /level_from)  
    for i in len(information):
        patch_object = patch_num/information[i][6] #information[i][6] = number of objects on the image
        convex_hull = np.load(result_addrs + information[i][0]) #read the convex_hull for the i-th image to check the convex hull coverage
        gray_scale_image = np.load(result_addrs + file_name + '_grayscale') #read the gray scale image for the i-th image to check the background color
        objct = read_openslide(information[i][0],information[i][7], result_addrs, level_from, level_to)
        for j in range(0,information[i][6]):
            extracted_patches = 0
            while extracted_patches < patch_object:
                x = randint(information[i][7][0],information[i][7][1]-mini_patch_size) #produce a random number inside the object's rectangle, width
                y = randint(information[i][7][2],information[i][7][3]-mini_patch_size) #produce a random number inside the object's rectangle, height
                mini_patch = convex_hull[x:x+mini_patch_size, y:y+mini_patch_size]
                mini_patch_summation = mini_patch.sum
                #check_covexhull_coverage
                if mini_patch_summation == (mini_patch_size^2)* 255: #the whole matrix must be inside the convex hull
                    #check_threshhold_coverage
                    mini_gray_scale = gray_scale_image[x:x+mini_patch_size, y:y+mini_patch_size]
                    bg_check = 0
                    for w in range(0,mini_patch_size):
                        for h in range(0,mini_patch_size):
                            if mini_gray_scale[w][h] > information[i][5]: #information[i][1] = background color
                                bg_check +=1
                    
                    if bg_check == mini_patch_size^2:
                        #resize the mini patch to a real patch going from level_from to level_to 
                        randW_walk = randint(0,patch_size) #produce a random number
                        randH_walk = randint(0,patch_size) #produce a random number
                        patch = objct(x+randW_walk:x+randW_walk+patch_size, y+randH_walk:y+randH_walk+patch_size) #return a patch in level_to image
                                               
    return patch