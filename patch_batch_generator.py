## Importing the modules
import os
os.system('clear')

from random import randint
from numpy.matrix import sum

'''
inputs:
patch_num: number of patches from each image we want to select for train and validation or test
information: file with information of images: {number of objects, objects positions, threshhold}
result_addrs: the address to read convexhulls and ...
level_from: the level to extract small and grayscale patches
level_to: to conver the biggest image
'''

def patches_for_batches(patch_num, information, result_addrs, level_from, level_to, patch_size):
    mini_patch_size = patch_size / 2^(level_to /level_from)  
    for i in len(information):
        patch_object = patch_num/information[i][2] #information[i][2]  = number of objects on the image
        convex_hull = [] #read the convex_hull for the i th image to check the convex hull coverage
        gray_scale_image = [] #read the gray scale image  for the i th image to check the threshhold
        objct = read_openslide(information[i][0],(information[i][2], i, result_addrs, level_from, level_to, patch_size)
        for j in range(0,information[i][2]):
            extracted_patches = 0
            while extracted_patches < patch_object:
                x = randint(information[i][3][0],information[i][3][1]-mini_patch_size) #produce a random number inside the object's rectangle, width
                y = randint(information[i][3][2],information[i][3][3]-mini_patch_size) #produce a random number inside the object's rectangle, height
                mini_patch = convex_hull[x:x+mini_patch_size, y:y+mini_patch_size]
                mini_patch_summation = mini_patch.sum
                #check_covexhull_coverage
                if mini_patch_summation == (mini_patch_size^2)* 255: #the whole matrix must be inside the convex hull
                    #check_threshhold_coverage
                    mini_gray_scale = gray_scale_image[x:x+mini_patch_size, y:y+mini_patch_size]
                    threshhold_check = 0
                    for w in range(0,mini_patch_size):
                        for h in range(0,mini_patch_size):
                            if mini_gray_scale[w][h] > information[i][1]: #information[i][1] = threshhlold
                                threshhold_check +=1                                
                    
                    if threshhold_check == mini_patch_size^2:
                        #resize the mini patch to a real patch going from level_from to level_to 
                        randW_walk = randint(0,patch_size) #produce a random number
                        randH_walk = randint(0,patch_size) #produce a random number
                        patch = objct(x+randW_walk:x+randW_walk+patch_size, y+randH_walk:y+randH_walk+patch_size) #return a patch in level_to image 
                        
##read the open slide
def read_openslide(information[i][0],information[i][2], i, result_addrs, level_from, level_to, patch_size):
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

