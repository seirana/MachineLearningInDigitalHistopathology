## Importing the modules
import os
os.system('clear')

##read the open slide
def read_openslide(image_name '''information[i][0]''', objct_list'''information[i][7]''', i_th_objct'''i''', result_addrs, level_from, level_to, patch_size):
    tmp_mat = gray_scale[objct_list[n][0]:objct_list[n][1]+1, objct_list[n][2]:objct_list[n][3]+1]
    resized_matrix = np.repeat(np.repeat(tmp_mat, resize_, axis=0), resize_, axis=1)
    del(tmp_mat)
    
    ##read the object in the max. magnification level
    heigth_ = int((objct_list[n][1]- objct_list[n][0]+1)*resize_)
    width_ = int((objct_list[n][3]- objct_list[n][2]+1)*resize_)
    leftup_i = int(objct_list[n][0]*resize_)
    leftup_j = int(objct_list[n][2]*resize_)
    
    if heigth_ * width_  < 2**28:
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
    return objct #it must be a 3D matrix   