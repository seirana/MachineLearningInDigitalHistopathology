import os
import openslide
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_isodata
from skimage.morphology import opening, closing, disk, dilation, convex_hull_image

img_addrs = "/home/seirana/NDPI/MaLTT/Immunohistochemistry/"
result_addrs = "/home/seirana/Desktop/Workstation/casp3/"

information = list()
pixel_info = list()
for file in sorted(os.listdir(img_addrs)):
    if file.endswith(".ndpi") and (file.find("Casp3") > -1 or file.find("casp3") > -1):        
        info_gathering = list()
        file_name = file.replace('.ndpi', '')
        info_gathering.append(file_name)
        print(file_name)
        
        slide = openslide.OpenSlide(img_addrs + file)
        levelCount = slide.level_count 
        info_gathering.append(levelCount)
        levelDimension = slide.level_dimensions          
        info_gathering.append(levelDimension)
        
        # read the image in desired level, convert it to gray scale and save it as a .jpeg file
        for i in range(0, levelCount):
            if levelDimension[levelCount-i-1][0] > 2000 and levelDimension[levelCount-i-1][1] > 1000:
                x = levelDimension[levelCount-i-1] 
                break
            
        img_th = levelCount - i-1
        img = slide.get_thumbnail(levelDimension[img_th])        
        image = np.array(img)
        info_gathering.append(img_th)
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(result_addrs + file_name + '_(1)gray_scale.jpg', gray_scale)
        org_gray_scale = np.copy(gray_scale)
       
        arr = np.shape(gray_scale)
        pixel = np.zeros(shape=(1, 256))
        for i in range(0, arr[0]):
            for j in range(0, arr[1]):
                pixel[0][gray_scale[i][j]] += 1
    
        back_ground = max(pixel[0, :])
        for i in range(0, 256):
            if pixel[0][i] == back_ground:
                bg_pixel = i
                break
            
        for i in range(0, arr[0]):
            for j in range(0, arr[1]):
                if gray_scale[i][j] > 230:
                    gray_scale[i][j] = bg_pixel  # to change too much light pixels to the color of background
                if gray_scale[i][j] < 50:
                    gray_scale[i][j] = bg_pixel  # to change too much dark pixels to the color of background
                  
        # find and apply isodata threshold on the image
        correct_thresh = False        
        while not correct_thresh:
            # gray_scale = np.copy(org_gray_scale)
            iso_thresh = threshold_isodata(gray_scale)            
            arr = np.shape(gray_scale) 
            
            pixel = np.zeros(shape=(1, 256))
            for i in range(0, arr[0]):
                for j in range(0, arr[1]):
                    pixel[0][gray_scale[i][j]] += 1
        
            for i in range(0, 56):
                pixel[0][i] = pixel[0][i]/arr[0]*arr[1]
        
            back_ground = max(pixel[0, :])
            
            mn = 0
            for i in range(0, iso_thresh):
                mn += pixel[0][i]
            mx = 0
            for i in range(iso_thresh, 255):
                mx += pixel[0][i]    
                
            sm = mn+mx
            pixel_info.append([file, iso_thresh, mn/sm, mx/sm, back_ground])
            
            for i in range(0, arr[0]):
                for j in range(0, arr[1]):
                    if gray_scale[i][j] >= iso_thresh:
                        gray_scale[i][j] = 255
                    else: 
                        gray_scale[i][j] = 0 
           
            # apply binary morphological filters on the image(opening, dilation, closing) and save it
            selem = disk(6) 
            opened = opening(gray_scale, selem)
            dilated = dilation(opened, selem)
            closed = closing(dilated, selem)
               
            tst = []
            for j in range(0, arr[1]):
                tst.append(0)
                for i in range(0, arr[0]):
                    tst[j] = tst[j] + gray_scale[i][j]     
                    
            sz = int(0.1 * arr[1])
            tmp = 0
            rec_ = 0
            for j in range(0, arr[1]-sz):
                for i in range(0, sz+1):
                    tmp = tmp + tst[j+i]
                    
                if tmp == 0:
                    rec_ = 1
                    break
                
            if rec_ == 1:    
                for i in range(0, levelDimension[img_th][1]):
                    for j in range(0, levelDimension[img_th][0]):
                        if org_gray_scale[i][j] >= iso_thresh:
                            org_gray_scale[i][j] = iso_thresh-1
            else:
                correct_thresh = True                
                      
        gray_scale = np.copy(closed) 
        cv2.imwrite(result_addrs + file_name + '_(2)edited_isodata.jpg', gray_scale)
                     
        # find boundries of the objects in the image ans save it
        mat = np.copy(gray_scale)
        for i in range(0, levelDimension[img_th][1]):
            for j in range(0, levelDimension[img_th][0]):
                if gray_scale[i][j] == 0:
                    mat[i][j] = 255  
                else:
                    mat[i][j] = 0
                    
        for i in range(1, levelDimension[img_th][1]-1):
            for j in range(1, levelDimension[img_th][0]-1):
                if gray_scale[i][j] == 0:
                    sm = 0
                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            sm = sm + gray_scale[i+k][j+l]
                            
                    if sm == 0:
                        mat[i][j] = 0
                    else:
                        mat[i][j] = 255   
        
        gray_scale = np.copy(mat)            
        cv2.imwrite(result_addrs + file_name + '_(3)boundries.jpg', gray_scale)

        # calculate the number of contours on the image and omit the small ones and save it
        ret, thresh = cv2.threshold(gray_scale, 127, 255, 0)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for i in range(0, len(contours)):
            mini = max(levelDimension[img_th][0], levelDimension[img_th][1])
            maxi = 0
            
            for j in range(0, len(contours[i])):
                if contours[i][j][0][0] > maxi:
                    maxi = contours[i][j][0][0]
                if contours[i][j][0][0] < mini:
                    mini = contours[i][j][0][0]
                    
            numH = maxi-mini+1             
            
            minj = max(levelDimension[img_th][0], levelDimension[img_th][1])
            maxj = 0
    
            for j in range(0, len(contours[i])):
                if contours[i][j][0][1] > maxj:
                    maxj = contours[i][j][0][1]
                if contours[i][j][0][1] < minj:
                    minj = contours[i][j][0][1]
                    
            numW = maxj-minj+1       
     
            tr = levelDimension[img_th][0] * levelDimension[img_th][1]
            if numH*numW < 0.001 * tr:
                for m in range(mini, maxi+1):
                    for n in range(minj, maxj+1):                    
                        gray_scale[n][m] = 0 
                    
            if numH*numW < 0.01 * tr:
                if len(contours[i]) < 0.001:
                    for m in range(mini, maxi+1):
                        for n in range(minj, maxj+1):                    
                            gray_scale[n][m] = 0   
                  
        cv2.imwrite(result_addrs + file_name + '_(4)omited_small_contours.jpg', gray_scale)
                
        # trace the image to find the object zones (sweep the columns)
        sz = np.shape(gray_scale)
        objct = []
        num -= 1
        cnt_pre = [0, 0, 0]
        cnt = 0
    
        for j in range(0, sz[1]):
            tmp = cnt_pre
            cnt_pre = [tmp[1], tmp[2], cnt]
            cnt = 0
            for i in range(0, sz[0]):
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
                if cnt_pre[0] == 1 and cnt_pre[1] == 0 and cnt_pre[2] == 0:
                    if (objct[num][1] - objct[num][0]) < (0.05 * sz[1]):
                        for k in range(objct[num][0], objct[num][1]+1):
                            for t in range(0, sz[0]):
                                gray_scale[t][k] = 0 
    
                        del objct[num]
                        num = num - 1
    
        # trace the image to find the object zones (sweep the rows)
        objct_list = list()     
        objct_num = -1
        if num > -1:
            for n in range(0, num+1):
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
                        if (cnt == 0 and cnt_pre[0] == 1 and cnt_pre[1] == 0 and cnt_pre[2] == 0) \
                                or (cnt == 1 and i > sz[0]-3):
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
            
                                        if find:
                                            break
            
                                    find = False    
                                    for mi in range(objct_list[objct_num][3], strt-1, -1):
                                        nd = objct_list[objct_num][3]                                
                                        for ni in range(objct_list[objct_num][0], objct_list[objct_num][1]+1):
                                            if gray_scale[ni][mi] == 255:
                                                nd = mi
                                                find = True
                                                break
            
                                        if find:
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
            print("The search was not successfull to find any object for this file: ", file)
            continue
                                     
        # drow a convex hall around each object and save it
        for k in range(0, objct_num):
            mat = np.zeros(shape=(levelDimension[img_th][1], levelDimension[img_th][0]))
            
            for i in range(objct_list[k][0], objct_list[k][1]+1):
                for j in range(objct_list[k][2], objct_list[k][3]+1):
                    mat[i][j] = gray_scale[i][j]
                    
            chull = convex_hull_image(mat)
                                
            for i in range(objct_list[k][0], objct_list[k][1]+1):
                for j in range(objct_list[k][2], objct_list[k][3]+1):
                    if not chull[i][j]:
                        gray_scale[i][j] = 0
                    else:
                        gray_scale[i][j] = 255                            
                        
        cv2.imwrite(result_addrs + file_name + '_(5)convex_hull.jpg', gray_scale)
            
        information.append(np.asarray(info_gathering, dtype=object))

        plt.close('all')
    else:
        continue   
    
file_Name = result_addrs + "_SlidesInfo_list"
np.save(file_Name, np.asarray(information, dtype=object))
