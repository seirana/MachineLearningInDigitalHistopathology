import os
import openslide
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from PIL import Image
from skimage.filters import threshold_isodata
from skimage.morphology import opening, closing, disk, dilation, convex_hull_image
import matplotlib.patches as patches
import math 
import Image_Properties as IMG ##mine
#import Random_Rotation_Mirroring as RRM ##mine


img_addrs = "/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/"
result_addrs = "/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/Results/"

IMG.slide_properties(img_addrs, result_addrs)

img_info = list()
img_info.append(["FileName", "levelCount", "levelDimension", "objct_num", "objct_list", "resize", "vertical overlap", "horisental overlap"])

for file in sorted(os.listdir(img_addrs)):
    if file.endswith(".ndpi"):       
    
        file_name = file.replace('.ndpi','')
        print file_name
        slide = openslide.OpenSlide(img_addrs + file)
        levelCount = slide.level_count    
        levelDimension = slide.level_dimensions          
        
        ##read the image in desired level, convert it to gray scale and save it as a .jpeg file 
        for i in range(0,levelCount):
            if levelDimension[levelCount-i-1][0] > 2000 and levelDimension[levelCount-i-1][1] > 1000:
                x = levelDimension[levelCount-i-1] 
                break
            
        img_th = levelCount-i-1
        img = slide.get_thumbnail(levelDimension[img_th])        
        image = np.array(img)
        del(img)
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(result_addrs + file_name + '_(1)gray_scale.jpg', gray_scale)
        org_gray_scale = np.copy(gray_scale)
       

        ##find and apply isodata threshold on the image 
        correct_thresh = False        
        while correct_thresh == False:
            gray_scale = np.copy(org_gray_scale)             
            iso_thresh = threshold_isodata(gray_scale)            
            arr = np.shape(gray_scale) 
            
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
        cv2.imwrite(result_addrs + file_name + '_(2)edited_isodata.jpg', gray_scale) 
        del(opened)
        del(dilated)
        del(closed)              
                      

        ##find boundries of the objects in the image ans save it
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
        cv2.imwrite(result_addrs + file_name + '_(3)boundries.jpg', gray_scale)
        del(mat)
        
                        
        ##calculate the number of contours on the image and omit the small ones and save it  
        ret,thresh = cv2.threshold(gray_scale,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)                        

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
                  
        cv2.imwrite(result_addrs + file_name + '_(4)omited_small_contours.jpg', gray_scale)
                    
                
        ##trace the image to find the object zones (sweep the columns)
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
    
    
        ##trace the image to find the object zones (sweep the rows)
        objct_list = list()     
        objct_num = -1
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
        list_size = len(objct_list)
        if list_size < 1:
            print "The search was not successfull to find any object for this file: ", file            
            continue
               
                       
        ##drow a convex hall around each object and save it
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
                        
        cv2.imwrite(result_addrs + file_name + '_(5)convex_hull.jpg', gray_scale)         
        del(chull)
           
            
        ##define the new image size (resizing)
        #print "There are different magnification levels,"
        #for i in range(0, img_th): 
            #print "level ", i, ":", levelDimension[i]

        #inpt = raw_input("Please choose the desired manginifasion layer: ") 
        #inpt = str(random.randint(0, img_th-1))
        inpt = str(0)
        sm = 0
        for ch in inpt:
            sm = sm + ord(ch)
            
        ln = 0
        for ch in str(img_th-1):
            ln = ln + ord(ch)
            
        while sm < 48 or sm > ln:        
            #inpt = raw_input("Please insert the desired manginifasion layer: ") 
            #inpt = str(random.randint(0, img_th-1))
            inpt = str(0)
            sm = 0
            for ch in inpt:
                sm = sm + ord(ch)      
  
        resize_ = levelDimension[int(inpt)][0]/levelDimension[img_th][0]
        
        if type(resize_) != int:
            print "resize_ is not integer!"
            continue   
                
        max_pos_patch_sz = resize_ * (objct_list[0][1] - objct_list[0][0]+1)
        if max_pos_patch_sz > resize_ * (objct_list[0][3] - objct_list[0][2]+1):
            max_pos_patch_sz = resize_ * (objct_list[0][3] - objct_list[0][2]+1)
        for i in range(1, list_size):
            if max_pos_patch_sz > resize_ * (objct_list[i][1] - objct_list[i][0]+1):
                max_pos_patch_sz = resize_ * (objct_list[i][1] - objct_list[i][0]+1)
            if max_pos_patch_sz > resize_ * (objct_list[i][3] - objct_list[i][2]+1):
                max_pos_patch_sz = resize_ * (objct_list[i][3] - objct_list[i][2]+1)            
          
               
        ##define a patch size
        if max_pos_patch_sz > 24:
            mini = 10
        else:
            mini = 4
        #inpt = raw_input("Please insert the desired patch size:(between", mini, "and", max_pos_patch_sz)    
        #inpt = str(random.randint(mini, max_pos_patch_sz))
        if max_pos_patch_sz > 100:
            inpt = str(100)
        else:
            print "patch size is: ", max_pos_patch_sz
            continue
        
        sm = 0
        for ch in inpt:
            sm = sm + ord(ch)
            
        ln1 = 0        
        for ch in str(mini):
            ln1 = ln1 + ord(ch)
        
        ln2 = 0        
        for ch in str(max_pos_patch_sz):
            ln2 = ln2 + ord(ch)
            
        while sm < ln1 or sm > ln2:        
            #inpt = raw_input("Please insert the desired patch size:(between", mini, "and", max_pos_patch_sz) 
            #inpt = str(random.randint(mini, max_pos_patch_sz))
            inpt = str(100)
            sm = 0
            for ch in inpt:
                sm = sm + ord(ch) 
                
        patch_size = int(inpt)        
        
          
        ##overlap size for the patches(horisental and vertical)
        #inpt = raw_input("Please insert the horisental_overlaping percentage:(between 0 and 99) ")
        #inpt = str(random.randint(0, 99))
        inpt = str(0)
        sm = 0
        for ch in inpt:
            sm = sm + ord(ch)
            
        ln1 = 0        
        for ch in str(0):
            ln1 = ln1 + ord(ch)
        
        ln2 = 0        
        for ch in str(99):
            ln2 = ln2 + ord(ch)
            
        while sm < ln1 or sm > ln2:        
            #inpt = raw_input("Please insert the horisental_overlaping percentage:(between 0 and 99) ") 
            #inpt = str(random.randint(0,99))
            inpt = str(0)
            sm = 0
            for ch in inpt:
                sm = sm + ord(ch)                 
        
        hor_ovlap = int(int(inpt)*patch_size/100)
        
        if hor_ovlap == patch_size:
            hor_ovlap = hor_ovlap-1  
        
        #inpt = raw_input("Please insert the vertical_overlaping percentage:(between 0 and 99) ")
        inpt = str(random.randint(0,99))
        inpt = str(0)
        sm = 0
        for ch in inpt:
            sm = sm + ord(ch)
            
        ln1 = 0        
        for ch in str(0):
            ln1 = ln1 + ord(ch)
        
        ln2 = 0        
        for ch in str(99):
            ln2 = ln2 + ord(ch)
            
        while sm < ln1 or sm > ln2:        
            #inpt = raw_input("Please insert the vertical_overlaping percentage:(between 0 and 99) ") 
            #inpt = str(random.randint(0, 99))
            inpt = str(0)
            sm = 0
            for ch in inpt:
                sm = sm + ord(ch)         
        
        ver_ovlap = int(int(inpt)*patch_size/100)        
           
        if ver_ovlap == patch_size:
            ver_ovlap = ver_ovlap-1

        
        ##trace objects of the image based on the patches        
        per_ = 80 / 100 * 255 ##minimum coverage of the convex hull by the patches, 255 is for white pixles
        patch_list = list()
        patch_info = list()
        
        cntr = 0
        for n in range(0, objct_num): 
            
            ##resize the object(s) from img_th to desired level
            tmp_mat = gray_scale[objct_list[n][0]:objct_list[n][1]+1, objct_list[n][2]:objct_list[n][3]+1]            
            resized_matrix = np.repeat(np.repeat(tmp_mat, resize_, axis=0), resize_, axis=1) 
            del(tmp_mat)

            ##read the object in the max. magnification level
            heigth_ = (objct_list[n][1]- objct_list[n][0]+1)*resize_ 
            width_ = (objct_list[n][3]- objct_list[n][2]+1)*resize_ 
            leftup_i  = objct_list[n][0]*resize_
            leftup_j  = objct_list[n][2]*resize_
            
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
                        patch_list.append(tmp)
                        patch_info.append([n, i ,j])         

        ##save the patch list to a file
        file_Name = result_addrs + file_name + "_patch_list"
        np.save(file_Name, patch_list)                  
        file_Name = result_addrs + file_name + "_patch_info"
        np.save(file_Name, patch_info) 
        del(tmp)
        del(patch_list)  
        del(patch_info)
        del(resized_matrix)
        del(max_mag_lev_obj)
        img_info.append([file_name, levelCount, levelDimension, img_th, objct_num, objct_list, resize_, patch_size, ver_ovlap, hor_ovlap])
        plt.close('all')
        
                
##save the information for the the images in a file
file_Name = result_addrs + "Images_INFO"
np.save(file_Name, img_info)

#patch_ = RRM.random_rot_mirr(patch_)

## in progress
#class get_patch:    
    
    ###showing the position of the patch in the jpeg image of the .ndpi file in 
    #ax.imshow(image)
    ## Create a Rectangle patch
    #rect = patches.Rectangle((patch_list[0]/resize_,patch_list[1]/resize_), patch_size/resize_, patch_size/resize_, linewidth=1, edgecolor='r', facecolor='none')
    ## Add the patch to the Axes
    #ax.add_patch(rect)
    ##plt.show()
    #return 