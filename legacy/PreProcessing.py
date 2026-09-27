import os
os.system('clear')

import openslide
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_isodata
from skimage.morphology import opening, closing, disk, dilation, convex_hull_image

def Preprocessing(img_addrs, result_addrs, file, information):
    
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
    def Build_jpeg_image(information, info_gathering):
        if levelCount >= 6:
            img_th = 5 # extract grayscale images of level 5th slides, Max level = level 0
            img = slide.get_thumbnail(levelDimension[img_th])
            image = np.array(img)
            info_gathering.append(img_th)
            
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
    
            cv2.imwrite(result_addrs + file_name + '_(2)normalized_gray_scale.jpg', gray_scale)
            plt.close('all')
            information = isodata(information, info_gathering, gray_scale, img_th, org_gray_scale, arr)
            return information
            
        else:
            return information
           
    '''
    define a new function
    '''
    ##find and apply isodata threshold on the image 
    def isodata(information, info_gathering, gray_scale, img_th, org_gray_scale, arr):
        #rearches for two-color backgrounds if still there are and omit them 
        correct_thresh = False
        while correct_thresh == False:
            iso_thresh = threshold_isodata(gray_scale)
           
            #make a coly of gray scale with two colors, 0 & 255
            check_rec = np.copy(gray_scale) 
            for i in range(0,arr[0]):
                for j in range(0,arr[1]):
                    if check_rec[i][j] >= iso_thresh:
                        check_rec[i][j] = 255
                    else:
                        check_rec[i][j] = 0
                        
            tst = []
            for j in range(0,arr[1]):
                tst.append(0)
                for i in range(0,arr[0]):
                    tst[j] = tst[j] + check_rec[i][j]
                    
            sz = int(0.1 * arr[1])
            tmp = 0
            rec_ = 0
            for j in range(0,arr[1]-sz):
                for i in range(0,sz+1):
                    tmp = tmp + tst[j+i]
                    
                if tmp == 0:
                    rec_ = 1
                    break
            
            #change the color of lighter background to the darker one
            if rec_ == 1:
                for i in range(0,levelDimension[img_th][1]):
                    for j in range(0,levelDimension[img_th][0]):
                        if gray_scale[i][j] >= iso_thresh:
                            gray_scale[i][j]  = iso_thresh-1
            else:
                correct_thresh = True
                
        iso_thresh = threshold_isodata(gray_scale) #calculate the new threshhold for the gray scale image
        
        #make a copy of gray scale, with two colors, 0 & 255
        for i in range(0,arr[0]):
            for j in range(0,arr[1]):
                if gray_scale[i][j] >= iso_thresh:
                    gray_scale[i][j] = 255
                else:
                    gray_scale[i][j] = 0
      
        pixel = np.zeros(shape=(1,256))
        for i in range(0,arr[0]):
            for j in range(0,arr[1]):
                pixel[0][gray_scale[i][j]] += 1
           
        back_ground = max(pixel[0,:])
        for i in range(0,256):
            if pixel[0][i] == back_ground:
                bg_pixel = i
                break 
            
        ##apply binary morphological filters on the image(opening, dilation, closing) and save it
        selem = disk(6)
        opened = opening(gray_scale, selem)
        dilated = dilation(opened, selem)
        closed = closing(dilated, selem)          
        gray_scale = np.copy(closed)
        
        cv2.imwrite(result_addrs + file_name + '_(3)edited_isodata.jpg', gray_scale)
        np.save(result_addrs + file_name + '_grayscale', gray_scale)
        plt.close('all')
        info_gathering.append(iso_thresh)
        info_gathering.append(bg_pixel)
        
        ##calling the next function
        information = boundries(information, info_gathering, gray_scale, img_th)
        return information
   
    '''
    define a new function
    '''
    ##find boundries of the objects in the image and save it
    def boundries(information, info_gathering, gray_scale, img_th):
        #change black to white & white to black in the gray scale image
        mat = np.copy(gray_scale)
        for i in range(0,levelDimension[img_th][1]):
            for j in range(0,levelDimension[img_th][0]):
                if gray_scale[i][j] == 0:
                    mat[i][j] = 255
                else:
                    mat[i][j] = 0
                    
        #drow boundries around the objects
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
        plt.close('all')
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
        new_lst = np.dot(objct_list,2^img_th)
        info_gathering.append(new_lst)
        list_size = len(objct_list)
        if list_size < 1:
            print ("The search was not successfull to find any object for this file: ", file_name)
        else:
            ##calling the next function    
            covex_hall(info_gathering, gray_scale, img_th, objct_list, objct_num, list_size) 
        
        information.append(np.asarray(info_gathering,dtype= object))
        return information
     
    '''
    define a new function
    '''
    ##drow a convex hall around each object and save it
    def covex_hall(info_gathering, gray_scale, img_th, objct_list, objct_num, list_size):
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
        resize_ = 2^img_th 
        resized_matrix = np.repeat(np.repeat(gray_scale, resize_, axis=0), resize_, axis=1)
        np.save(result_addrs + file_name + "_convex_hull", resized_matrix)
        plt.close('all')
        return
    
    information = Build_jpeg_image(information, info_gathering)
    return information   