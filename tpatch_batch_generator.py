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

def train_patches_for_batches(addrs, img_lst, img_size, patch_size, rand_list, tt):             
    for n in range(0, objct_num): 
        
        ##resize the object(s) from img_th to desired level
        tmp_mat = gray_scale[objct_list[n][0]:objct_list[n][1]+1, objct_list[n][2]:objct_list[n][3]+1]            
        resized_matrix = np.repeat(np.repeat(tmp_mat, resize_, axis=0), resize_, axis=1) 
        del(tmp_mat)
    
        ##read the object in the max. magnification level
        heigth_ = int((objct_list[n][1]- objct_list[n][0]+1)*resize_) 
        width_ = int((objct_list[n][3]- objct_list[n][2]+1)*resize_) 
        leftup_i  = int(objct_list[n][0]*resize_)
        leftup_j  = int(objct_list[n][2]*resize_)
        
        if heigth_  * width_  < 2**28:
            tmp_img = slide.read_region((leftup_j, leftup_i), 0, (width_, heigth_))
            max_mag_lev_obj = np.array(tmp_img)
            del(tmp_img)
        else:
            x_ = int(math.ceil(heigth_*width_/2**28))  
            new_heigth_ = int(math.ceil(heigth_/x_))
            tmp_img = slide.read_region((leftup_j, leftup_i), 0, (width_, new_heigth_))
            max_mag_lev_obj = np.array(tmp_img) 
            del(tmp_img)
            
            if x_ > 2:
                for i in range(1,x_-1):
                    tmp_img = slide.read_region((leftup_j, leftup_i+new_heigth_*i), 0, (width_, new_heigth_))
                    t_mat= np.array(tmp_img)
                    max_mag_lev_obj = np.append(max_mag_lev_obj, t_mat, axis = 0)
       
            tmp_img = slide.read_region((leftup_j, leftup_i+new_heigth_*(x_-1)), 0, (width_, heigth_-new_heigth_*(x_-1)))
            t_mat= np.array(tmp_img)
            max_mag_lev_obj = np.append(max_mag_lev_obj, t_mat, axis = 0)
                    
        for i in range(0,int((heigth_ - ver_ovlap) / (patch_size - ver_ovlap))):
            for j in range(0,int((width_ - hor_ovlap) / (patch_size - hor_ovlap))):
                tmp = max_mag_lev_obj[(patch_size - ver_ovlap)*i:(patch_size - ver_ovlap)*i+patch_size,(patch_size - hor_ovlap)*j:(patch_size - hor_ovlap)*j+patch_size]
                
                #check this line  ##check the coverage condition for the patch, to make all patches in the min. possible time
                summation = np.sum(resized_matrix[(patch_size - ver_ovlap)*i:(patch_size - ver_ovlap)*i+patch_size,(patch_size - hor_ovlap)*j:(patch_size - hor_ovlap)*j+patch_size])
                if summation <= per_ * (patch_size ** 2):
                    break
                else:
                    if len(patch_list) > ((2**31)/((patch_size**2)*3)):#len(patch_list) > ((5*(10**8))/(patch_size**2)): #images are RGBA(4D) #the size of the matrix should be less than 4GB
                        file_Name = result_addrs + file_name + "_patch_list" + str(ls)
                        np.save(file_Name, np.asarray(patch_list,dtype= object))
                        print(np.shape(patch_list))
                        ls += 1
                        sz_list += len(patch_list)
                        patch_list = list()
                        alpha_remov = tmp[:,:,0:3] #remove alpha channel
                        patch_list.append(alpha_remov)
                    
                    
                    alpha_remov = tmp[:,:,0:3]
                    patch_list.append(alpha_remov) #remove alpha chnnel
                    patch_info.append([n, objct_list[n][0]*resize_+(patch_size - ver_ovlap)*i, objct_list[n][2]*resize_+(patch_size - hor_ovlap)*j])

    return patch_list