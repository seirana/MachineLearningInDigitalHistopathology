## Importing the modules
import os
os.system('clear')

import numpy as np
import math
import openslide
from Random_Rotation_mirroring import random_rot_mirr

##read the open slide
def read_openslide(addrs, slide_ID, leftup_i, leftup_j, patch_size):
    slide = openslide.OpenSlide(addrs + slide_ID)

    ##read the object in the max. magnification level
    heigth_ = patch_size
    width_ = patch_size
    
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
   
    tmp_img = slide.read_region((leftup_j, leftup_i+new_heigth_*(x_-1)), 0, (width_, heigth_-new_heigth_*(x_-1)))
    t_mat= np.array(tmp_img)
    max_mag_lev_obj = np.append(max_mag_lev_obj, t_mat, axis = 0) 
    patch = max_mag_lev_obj[:,:,0] #remove the alpha channel
    #mirror and rotate the patch randomly
    patch = random_rot_mirr(patch)
    
    return patch #it must be a 3D matrix   