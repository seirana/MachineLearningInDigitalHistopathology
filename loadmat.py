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
import slide_properties


ndpi_addrs = "/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/"
result_addrs = "/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/Results/"
img_info = list()
img_info.append(["FileName", "levelCount", "levelDimension", "objct_num", "objct_list", "resize", "vertical overlap", "horisental overlap"])
img_pro = list()
img_pro.append(["File name", "Level count", "Level dimension", "Quickhash", "Objective power", "Vendor", "Bounds color", "comment", "MAPP X", "MPP Y", "Bounds X", "Bounds Y", "Bounds width", "Bounds height"])

for file in sorted(os.listdir(ndpi_addrs)):
    if file.endswith(".ndpi"): 
        file_name = file.replace('.ndpi','')
        
        ##open the .ndpi file and find the level_count  and level dimmensions and other attributes of the image (if known)     
        def slide_properties(slide_name):
            
            slide = openslide.OpenSlide(ndpi_addrs + file)
            levelCount = slide.level_count    
            levelDimension = slide.level_dimensions
            w0 = slide.properties.get(openslide.PROPERTY_NAME_QUICKHASH1)
            w1 = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
            w2 = slide.properties.get(openslide.PROPERTY_NAME_BACKGROUND_COLOR)
            w3 = slide.properties.get(openslide.PROPERTY_NAME_VENDOR)
            w4 = slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_WIDTH)
            w5 = slide.properties.get(openslide.PROPERTY_NAME_COMMENT)
            w6 = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
            w7 = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
            w8 = slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_X)        
            w9 = slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_Y)        
            w10 = slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_WIDTH)        
            w11 = slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_HEIGHT)
            
            img_pro.append([file_name, levelCount, levelDimension, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11])
            return
        
        
        # should go to the end of the file 
        np.savetxt("/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/Results/AAA.csv", img_pro, fmt='%s')   

    
    
    
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
            tmp = 0;
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
    
                if cnt == 1:
                    if cnt_pre[0] + cnt_pre[1] + cnt_pre[2] == 0:                        
                        objct_list.append([i, i, objct[n][0], objct[n][1]]) 
                        objct_num = objct_num+1
    
                    else:
                        nc = objct_list.pop()
                        objct_list.append([nc[0], i, nc[2], nc[3]])
    
                else:
                    if cnt_pre[0] == 1 and cnt_pre[1] == 0 and cnt_pre[2] == 0:                           
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
           
            
        ##resize the image
        print "There are different magnification levels,"
        for i in range(0, img_th): 
            print "level ", i, ":", levelDimension[i]

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
            if max_pos_patch_sz > resize_ * (objct_list[0][1] - objct_list[0][0]+1):
                max_pos_patch_sz = resize_ * (objct_list[0][1] - objct_list[0][0]+1)
            if max_pos_patch_sz > resize_ * (objct_list[0][3] - objct_list[0][2]+1):
                max_pos_patch_sz = resize_ * (objct_list[0][3] - objct_list[0][2]+1)            
            
               
        
        ###define a patch size
        if max_pos_patch_sz > 24:
            mini = 10
        else:
            mini = 4
        #inpt = raw_input("Please insert the desired patch size:(between", mini, "and", max_pos_patch_sz)    
        #inpt = str(random.randint(mini, max_pos_patch_sz))
        inpt = str(100)
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
        
        
          
        ###overlap size for the patches
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
        per_ = 80 / 100 ##minimum coverage of the convex hull by the patches
        patch_list = list()
        for n in range(0, objct_num): 
            
            ##resize the object from img_th to desired level
            tmp_mat = np.zeros(objct_list[n][1]- objct_list[n][0], objct_list[n][3]- objct_list[n][2])
            for i in range(0, objct_list[n][1]- objct_list[n][0]):
                for j in range(0, objct_list[n][3]- objct_list[n][2]):
                    tmp_mat[i][j] = gray_scale[objct_list[n][0]+i][objct_list[n][2]+j]
            
            resized_matrix = np.repeat(np.repeat(tmp_mat, resize_, axis=0), resize_, axis=1) 
            
            
            ##read the object in the max. magnification level
            tmp_img = slide.read_region((objct_list[n][0]*resize_,objct_list[n][2]*resize_), 0, ((objct_list[n][1]- objct_list[n][0])*resize_, (objct_list[n][3]- objct_list[n][2])*resize_))
            max_mag_lev_obj = np.array(tmp_img)
        
                        
            for i in range(0,int(((objct_list[n][1] - objct_list[n][0]) * resize_ - ver_ovlap) / (patch_size - ver_ovlap))):
                for j in range(0,int(((objct_list[n][3] - objct_list[n][2]) * resize_ - hor_ovlap) / (patch_size - hor_ovlap))):
                    w = 0
                    tmp = [[-1 for x in range(0,patch_size)] for y in range(0,patch_size)]
                    non_cov_patch = False
                    
                    for ip in range(i * (patch_size - ver_ovlap) + (objct_list[n][0] * resize_), i * (patch_size - ver_ovlap) + (objct_list[n][0] * resize_) + patch_size):
                        for jp in range(j * (patch_size - hor_ovlap) + objct_list[n][2], j * (patch_size - hor_ovlap) + objct_list[n][2] + patch_size):
                            tmp[ip % patch_size][jp % patch_size] = max_mag_lev_obj[ip][jp]
                            if resized_matrix[ip][jp] == 0:
                                w = w+1                                 
                                #check this line  ##check the coverage condition for the patch, to make all patches in the min. possible time                               
                                if (patch_size - (ip % patch_size)) * patch_size + (patch_size - (jp % patch_size))+ w <= per_ * patch_size * patch_size:
                                    non_cov_patch = True
                                    break
                                
                        if non_cov_patch == True:
                            break

                    if non_cov_patch == False and w >= per_ * patch_size * patch_size:
                        patch_list.append([n, i, j, tmp])
                        

                    
            
        ##save the patch list to a file
        file_Name = result_addrs + file_name + "_patch_list"
        fileObject = open(file_Name,'wb') 
        pickle.dump(patch_list,fileObject)   
        fileObject.close()
    
        ##save image information to the file
        img_info.append([file_name, levelCount, levelDimension, levelDimension[img_th], objct_num, objct_list, resize_, patch_size, ver_ovlap, hor_ovlap])
        plt.close('all')
        

        
##save the information for the the images in a file
file_Name = result_addrs + "Images_INFO"
fileObject = open(file_Name,'wb') 
pickle.dump(img_info,fileObject)   
fileObject.close()

class random_rot_mirr:

    ##rotate a patch randomly
    i = random.randint(0, 7)
    
    if i == 0:
        rand_rot_mirr = np.rot90(patch_, k=0)
    if i == 1:
        rand_rot_mirr = np.rot90(patch_, k=1)
    if i == 2:
        rand_rot_mirr = np.rot90(patch_, k=2)
    if i == 3:
        rand_rot_mirr = np.rot90(patch_, k=3)
    if i == 4:
        rand_rot_mirr = np.fliplr(np.rot90(patch_, k=0))
    if i == 5:
        rand_rot_mirr = np.fliplr(np.rot90(patch_, k=1))
    if i == 6:
        rand_rot_mirr = np.fliplr(np.rot90(patch_, k=2))
    if i == 7:
        rand_rot_mirr = np.fliplr(np.rot90(patch_, k=3))
        
    return rant_rot_mirr



# in progress
class get_patch:    
    
    ##showing the position of the patch in the jpeg image of the .ndpi file in 
    ax.imshow(image)
    # Create a Rectangle patch
    rect = patches.Rectangle((patch_list[0]/resize_,patch_list[1]/resize_), patch_size/resize_, patch_size/resize_, linewidth=1, edgecolor='r', facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    #plt.show()
    return 
    