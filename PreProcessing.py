import os
os.system('clear')

import openslide
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.filters import threshold_isodata
from skimage.morphology import opening, closing, disk, dilation, convex_hull_image

def Preprocessing(img_addrs, result_addrs, file, img_size, patch_size, ver_ovlap, hor_ovlap, per, information, pixel_info):
    
    
    slide = openslide.OpenSlide(img_addrs + file)
    file_name = file.replace('.ndpi','')
    levelCount = slide.level_count
    levelDimension = slide.level_dimensions
    
    info_gathering = list()    
    info_gathering.append(file_name)
    info_gathering.append(levelCount)
    info_gathering.append(levelDimension)
    
    
    '''
    define a new function
    '''
    ##read the image in desired level, convert it to gray scale and save it as a .jpeg file
    def Build_jpeg_image(information,pixel_info, info_gathering):
        for i in range(0,levelCount):
            if levelDimension[levelCount-i-1][0] > 2000 and levelDimension[levelCount-i-1][1] > 1000:
                break
            
        img_th = levelCount-i-1
        img = slide.get_thumbnail(levelDimension[img_th])
        image = np.array(img)
        info_gathering.append(img_th)
        del(img) 
        
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)                
        cv2.imwrite(result_addrs + file_name + '_(1)gray_scale.jpg', gray_scale)
        org_gray_scale = np.copy(gray_scale)
        arr = np.shape(gray_scale)
        pixel = np.zeros(shape=(1,256))
        for i in range(0,arr[0]):
            for j in range(0,arr[1]):
                pixel[0][gray_scale[i][j]] += 1
    
        back_ground = max(pixel[0,:])
        for i in range(0,256):
            if pixel[0][i] == back_ground:
                bg_pixel = i
                break
            
        for i in range(0,arr[0]):
            for j in range(0,arr[1]):
                if gray_scale[i][j] > 230:
                    gray_scale[i][j] = bg_pixel #to change too much light pixels to the color of background
                if gray_scale[i][j] < 50:
                    gray_scale[i][j] = bg_pixel #to change too much dark pixels to the color of background                     

        cv2.imwrite(result_addrs + file_name + '_(2)Normalized_gray_scale.jpg', gray_scale)
        information,pixel_info = isodata(information, pixel_info, info_gathering, gray_scale, img_th, org_gray_scale, arr)
        return information,pixel_info
       
       
    '''
    define a new function
    '''
    ##find and apply isodata threshold on the image 
    def isodata(information, pixel_info, info_gathering, gray_scale, img_th, org_gray_scale, arr):        
        correct_thresh = False
        while correct_thresh == False:
            iso_thresh = threshold_isodata(gray_scale)
            
            pixel = np.zeros(shape=(1,256))
            for i in range(0,arr[0]):
                for j in range(0,arr[1]):
                    pixel[0][gray_scale[i][j]] += 1
        
            for i in range(0,256):
                pixel[0][i] = pixel[0][i]/arr[0]*arr[1]
        
            back_ground = max(pixel[0,:])
            
            mn = 0
            for i in range(0,iso_thresh):
                mn += pixel[0][i]
            mx = 0
            for i in range(iso_thresh,255):
                mx += pixel[0][i]
                
            sm = mn+mx
            pixel_info.append([file_name, iso_thresh, mn/sm, mx/sm, back_ground])
            
            for i in range(0,arr[0]):
                for j in range(0,arr[1]):
                    if gray_scale[i][j] >= iso_thresh:
                        gray_scale[i][j] = 255
                    else:
                        gray_scale[i][j] = 0
            
            ##apply binary morphological filters on the image(opening, dilation, closing) and save it
            selem = disk(6) 
            opened = opening(gray_scale, selem)
            dilated = dilation(opened, selem)
            closed = closing(dilated, selem)
                        
            tst = []
            for j in range(0,arr[1]):
                tst.append(0)
                for i in range(0,arr[0]):
                    tst[j] = tst[j] + gray_scale[i][j]
                    
            sz = int(0.1 * arr[1])
            tmp = 0
            rec_ = 0
            for j in range(0,arr[1]-sz):
                for i in range(0,sz+1):
                    tmp = tmp + tst[j+i]
                    
                if tmp == 0:
                    rec_ = 1
                    break
                
            if rec_ == 1:
                for i in range(0,levelDimension[img_th][1]):
                    for j in range(0,levelDimension[img_th][0]):
                        if org_gray_scale[i][j] >= iso_thresh:
                            org_gray_scale[i][j]  = iso_thresh-1
            else:
                correct_thresh = True
                  
        gray_scale = np.copy(closed)
        cv2.imwrite(result_addrs + file_name + '_(3)edited_isodata.jpg', gray_scale)
        information, pixel_info = boundries(information, info_gathering, gray_scale, img_th)
        return information, pixel_info

   
    '''
    define a new function
    '''
    ##find boundries of the objects in the image and save it
    def boundries(information, info_gathering, gray_scale, img_th):
        mat = np.copy(gray_scale)
        for i in range(0,levelDimension[img_th][1]):
            for j in range(0,levelDimension[img_th][0]):
                if gray_scale[i][j] == 0:
                    mat[i][j] = 255
                else:
                    mat[i][j] = 0
                    
        for i in range(1,levelDimension[img_th][1]-1):
            for j in range(1,levelDimension[img_th][0]-1):
                if gray_scale[i][j] == 0:
                    sm = 0
                    for k in range(-1,2):
                        for l in range(-1,2):
                            sm = sm + gray_scale[i+k][j+l]
                            
                    if sm == 0:
                        mat[i][j] = 0
                    else:
                        mat[i][j] = 255
        
        gray_scale = np.copy(mat)
        cv2.imwrite(result_addrs + file_name + '_(4)boundries.jpg', gray_scale)
        ##calling the next function
        information = contours_counting(information, info_gathering, gray_scale, img_th)
        return information
        
        
    '''
    define a new function
    '''               
    ##calculate the number of contours on the image and omit the small ones and save it  
    def contours_counting(information, info_gathering, gray_scale, img_th):
        ret,thresh = cv2.threshold(gray_scale,127,255,0)
        _ , contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        

        for i in range(0,len(contours)):
            mini = max(levelDimension[img_th][0], levelDimension[img_th][1])
            maxi = 0
            
            for j in range(0,len(contours[i])):
                if contours[i][j][0][0] > maxi:
                    maxi = contours[i][j][0][0]
                if contours[i][j][0][0] < mini:
                    mini = contours[i][j][0][0]
                    
            numH = maxi-mini+1
            
            minj = max(levelDimension[img_th][0], levelDimension[img_th][1])
            maxj = 0
    
            for j in range(0,len(contours[i])):
                if contours[i][j][0][1] > maxj:
                    maxj = contours[i][j][0][1]
                if contours[i][j][0][1] < minj:
                    minj = contours[i][j][0][1]
                    
            numW = maxj-minj+1
     
            tr = levelDimension[img_th][0] * levelDimension[img_th][1]
            if numH*numW < 0.001 * tr:
                for m in range(mini,maxi+1):
                    for n in range(minj, maxj+1):
                        gray_scale[n][m] = 0
                    
            if numH*numW < 0.01 * tr:
                if len(contours[i]) < 0.001:
                    for m in range(mini,maxi+1):
                        for n in range(minj, maxj+1):
                            gray_scale[n][m] = 0
                  
        cv2.imwrite(result_addrs + file_name + '_(5)omited_small_contours.jpg', gray_scale)
        ##calling the next function
        information = object_zones_horizontal(information, info_gathering, gray_scale, img_th)
        return information
    
    '''
    define a new function
    '''       
    ##trace the image to find the object zones (sweep the columns)
    def object_zones_horizontal(information, info_gathering, gray_scale, img_th):
        sz = np.shape(gray_scale)
        objct = []
        num = -1
        cnt_pre = [0, 0, 0]
        cnt = 0
    
        for j in range(0, sz[1]):
            tmp = cnt_pre
            cnt_pre = [tmp[1], tmp[2], cnt]
            cnt = 0
            for i in range(0,sz[0]):
                if gray_scale[i][j] == 255:
                    cnt = 1
                    break
    
            if cnt == 1:
                if cnt_pre[0] + cnt_pre[1] + cnt_pre[2] == 0:
                    num = num+1
                    objct.append([j, j])
                else:
                    nc = objct.pop()
                    objct.append([nc[0], j])
    
            else:
                if cnt_pre[0] == 1 and cnt_pre[1]  == 0 and cnt_pre[2] == 0:
                    if (objct[num][1] - objct[num][0]) < (0.05 * sz[1]):
                        for k in range(objct[num][0], objct[num][1]+1):
                            for t in range(0,sz[0]):
                                gray_scale[t][k] = 0
    
                        del objct[num]
                        num = num - 1                        
        ##calling the next function                
        information = object_zones_vertical(information, info_gathering, gray_scale, objct, num, img_th)
        return information
        
    '''
    define a new function
    '''   
    ##trace the image to find the object zones (sweep the rows)         
    def object_zones_vertical(information, info_gathering, gray_scale, objct, num,  img_th):
        objct_list = list()
        objct_num = -1
        
        sz = np.shape(gray_scale)
        if num > -1:
            for n in range(0,num+1):
                cnt_pre = [0, 0, 0]
                cnt = 0
        
                for i in range(0, sz[0]):
                    tmp = cnt_pre
                    cnt_pre = [tmp[1], tmp[2], cnt]
                    cnt = 0
                    for j in range(objct[n][0], objct[n][1]+1):
                        if gray_scale[i][j] == 255:
                            cnt = 1
                            break
        
                    if cnt == 1 and i < sz[0]-4:
                        if cnt_pre[0] + cnt_pre[1] + cnt_pre[2] == 0:
                            objct_list.append([i, i, objct[n][0], objct[n][1]])
                            objct_num = objct_num+1
        
                        else:
                            nc = objct_list.pop()
                            objct_list.append([nc[0], i, nc[2], nc[3]])
        
                    else:
                        if (cnt == 0 and cnt_pre[0] == 1 and cnt_pre[1] == 0 and cnt_pre[2] == 0) or (cnt == 1 and i > sz[0]-3):
                            if objct_num > -1:
                                if (objct_list[objct_num][1] - objct_list[objct_num][0]) < (0.05 * sz[0]):
                                    for k in range(objct[n][0], objct[n][1]+1):
                                        for t in range(objct_list[objct_num][0], objct_list[objct_num][1]+1):
                                            gray_scale[t][k] = 0
            
                                    del objct_list[objct_num]
                                    objct_num = objct_num-1
            
                                else:
                                    find = False
                                    for mi in range(objct_list[objct_num][2], objct_list[objct_num][3]+1):
                                        strt = objct_list[objct_num][2]
                                        for ni in range(objct_list[objct_num][0], objct_list[objct_num][1]+1):
                                            if gray_scale[ni][mi] == 255:
                                                strt = mi
                                                find = True
                                                break
            
                                        if find == True:
                                            break
            
                                    find = False
                                    for mi in range(objct_list[objct_num][3], strt-1, -1):
                                        nd = objct_list[objct_num][3]
                                        for ni in range(objct_list[objct_num][0], objct_list[objct_num][1]+1):
                                            if gray_scale[ni][mi] == 255:
                                                nd = mi
                                                find = True
                                                break
            
                                        if find == True:
                                            break                                    
            
                                    if (nd - strt) < (0.05 * sz[1]):
                                        for t in range(objct_list[objct_num][0], objct_list[objct_num][1]+1):
                                            for k in range(objct_list[objct_num][2], objct_list[objct_num][3]+1):
                                                gray_scale[t][k] = 0
            
                                        del objct_list[objct_num]
                                        objct_num = objct_num-1

                                
        objct_num = objct_num+1
        info_gathering.append(objct_num)
        info_gathering.append(objct_list)
        list_size = len(objct_list)
        if list_size < 1:
            print ("The search was not successfull to find any object for this file: ", file_name)
            return information
        ##calling the next function    
        information =  covex_hall(information, info_gathering, gray_scale, img_th, objct_list, objct_num, list_size) 
        return information

        
    '''
    define a new function
    '''
    ##drow a convex hall around each object and save it
    def covex_hall(information, info_gathering, gray_scale, img_th, objct_list, objct_num, list_size):
        for k in range(0,objct_num):
            mat = np.zeros(shape=(levelDimension[img_th][1],levelDimension[img_th][0]))
            
            for i in range(objct_list[k][0], objct_list[k][1]+1):
                for j in range(objct_list[k][2], objct_list[k][3]+1):
                    mat[i][j] = gray_scale[i][j]
                    
            chull = convex_hull_image(mat)
                                
            for i in range(objct_list[k][0], objct_list[k][1]+1):
                for j in range(objct_list[k][2], objct_list[k][3]+1):
                    if chull[i][j] == False:
                        gray_scale[i][j] = 0
                    else:
                        gray_scale[i][j] = 255
                        
        cv2.imwrite(result_addrs + file_name + '_(6)convex_hull.jpg', gray_scale)
        ##calling the next function
        information = build_high_resolusion_image_and_extrace_patches(information, info_gathering, gray_scale, img_th, objct_list, objct_num, list_size)
        return information
    
    
    '''
    define a new function
    '''    
    ##define the new image size (resizing)
    def build_high_resolusion_image_and_extrace_patches(information, info_gatherin, gray_scale, img_th, objct_list, objct_num, list_size):
        patch_info = list()
        if img_size < img_th:
            print("selected dimension level is smaller than the bacis level dimension then resizeig minimizes the image size.")
            return information
        else:
            resize_ = levelDimension[int(img_size)][0]/levelDimension[img_th][0]
            
        info_gathering.append(resize_)
        
        if math.floor(resize_) < resize_:
            print ("resize_ is not integer!")
            return information
                
        max_pos_patch_sz = resize_ * (objct_list[0][1] - objct_list[0][0]+1)
        if max_pos_patch_sz > resize_ * (objct_list[0][3] - objct_list[0][2]+1):
            max_pos_patch_sz = resize_ * (objct_list[0][3] - objct_list[0][2]+1)
        for i in range(1, list_size):
            if max_pos_patch_sz > resize_ * (objct_list[i][1] - objct_list[i][0]+1):
                max_pos_patch_sz = resize_ * (objct_list[i][1] - objct_list[i][0]+1)
            if max_pos_patch_sz > resize_ * (objct_list[i][3] - objct_list[i][2]+1):
                max_pos_patch_sz = resize_ * (objct_list[i][3] - objct_list[i][2]+1)
                
        if patch_size >=  max_pos_patch_sz:
            print ("Patchsize must be smaller than:", max_pos_patch_sz)
            return information
        
        sz_list = 0
        patch_list = list()
        ls = 0
        for n in range(0, objct_num):
            
            ##resize the object(s) from img_th to desired level
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
                        
            for i in range(0,int((heigth_ - ver_ovlap) / (patch_size - ver_ovlap))):
                for j in range(0,int((width_ - hor_ovlap) / (patch_size - hor_ovlap))):
                    tmp = max_mag_lev_obj[(patch_size - ver_ovlap)*i:(patch_size - ver_ovlap)*i+patch_size,(patch_size - hor_ovlap)*j:(patch_size - hor_ovlap)*j+patch_size]
                    
                    summation = np.sum(resized_matrix[(patch_size - ver_ovlap)*i:(patch_size - ver_ovlap)*i+patch_size,(patch_size - hor_ovlap)*j:(patch_size - hor_ovlap)*j+patch_size])
                    if summation <= per * (patch_size ** 2):
                        break
                    else:
                        if len(patch_list) > ((2**31)/((patch_size**2)*3)):#len(patch_list) > ((5*(10**8))/(patch_size**2)): #images are RGBA(4D) #the size of the matrix should be less than 4GB
                            file_Name = result_addrs + file_name + "_patch_list" + str(ls)
                            np.save(file_Name, np.asarray(patch_list,dtype= object))
                            ls += 1
                            sz_list += len(patch_list)
                            patch_list = list()
                            alpha_remov = tmp[:,:,0:3] #remove alpha channel
                            patch_list.append(alpha_remov)
                        
                        
                        alpha_remov = tmp[:,:,0:3]
                        patch_list.append(alpha_remov) #remove alpha chnnel
                        patch_info.append([n, objct_list[n][0]*resize_+(patch_size - ver_ovlap)*i, objct_list[n][2]*resize_+(patch_size - hor_ovlap)*j])
                        
        ##save the patch list and info to the files
        file_Name = result_addrs + file_name + "_patch_list" + str(ls)     
        np.save(file_Name, np.asarray(patch_list,dtype= object))
        file_Name = result_addrs + file_name + "_patch_info"
        np.save(file_Name, np.asarray(patch_info,dtype= object))
        sz_list += len(patch_list)
        
        info_gathering.append(sz_list)
        information.append(np.asarray(info_gathering,dtype= object))
        plt.close('all')
        return information
    
    information, pixel_info = Build_jpeg_image(information, pixel_info, info_gathering)
    return information, pixel_info    