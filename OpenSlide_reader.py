## Importing the modules
import os
os.system('clear')

import numpy as np
import math
import openslide

##read the open slide
def read_openslide(file_name, objct_list, n, img_addrs, result_addrs):
    slide = openslide.OpenSlide(img_addrs + file_name)

    ##read the object in the max. magnification level
    heigth_ = int((objct_list[n][1]- objct_list[n][0]+1)) 
    width_ = int((objct_list[n][3]- objct_list[n][2]+1)) 
    leftup_i  = int(objct_list[n][0])
    leftup_j  = int(objct_list[n][2])
    
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
    objct = max_mag_lev_obj[:,:,0] #remove the alpha channel               
    return objct #it must be a 3D matrix   