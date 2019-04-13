import openslide
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.filters import try_all_threshold, threshold_isodata
from PIL import Image
from skimage.morphology import opening, closing, disk, dilation
from skimage.morphology import convex_hull_image, remove_small_objects
import pickle
import random



ndpi_addrs = "/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/"
result_addrs = "/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/Results/"
img_info = list()
img_info.append(["FileName", "levelCount", "levelDmension", "objct_num", "objct_list"])

for file in os.listdir(ndpi_addrs):
    if file.endswith(".ndpi"): 
        
        ##open the .ndpi file
        slide = openslide.OpenSlide(ndpi_addrs + file)
        
        #
        print file
        file = "07A-D_MaLTT_Ctrl242h_CD3_MaLTT_Ctrl42h_CD3_07A-D - 2015-07-08 17.11.49.ndpi"
        slide = openslide.OpenSlide(ndpi_addrs + file)
        
        ##find the level_count               
        levelCount = slide.level_count

        ##find the level_dimensions            
        levelDimension = slide.level_dimensions

        ##show image in the minimum level
        for i in range(0,levelCount):
            if levelDimension[levelCount-i-1][0] > 2000 and levelDimension[levelCount-i-1][1] > 1000:
                x = levelDimension[levelCount-i-1] 
                break
            
        img_th = levelCount-i-1
        img = slide.get_thumbnail(levelDimension[img_th])
       
        
        ##change the minimum-level image to gray-scale             
        image = np.array(img)
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        
        ##save the minimum-level gray-scale image as a .jpg file  
        cv2.imwrite(result_addrs + file + '_(1)gray_scale.jpg', gray_scale)
        
        #
        print np.shape(gray_scale)
        
        
        
        ##apply all thesolds in the minimum-level gray-scale image                   
        fig, ax = try_all_threshold(gray_scale, figsize=(10, 8), verbose=False)
        
        ##save  the results of all thesolds in the minimum-level gray-scale image as a .jpg file
        fig.savefig(result_addrs + file + '_(2)try_all_threshold.jpg')

        
                 
        ##apply isodata threshold on the minimum-level gray-scale image  
        thresh = threshold_isodata(gray_scale)
        binary = gray_scale> thresh  
        
        fig, axes = plt.subplots(ncols=2, figsize=(8, 3))
        ax = axes.ravel()
        
        ax[0].imshow(gray_scale, cmap=plt.cm.gray)
        ax[0].set_title('Original image')
        
        ax[1].imshow(binary, cmap=plt.cm.gray)
        ax[1].set_title('Result')
        
        for a in ax:
            a.axis('off')           
                 
             
        ##using isodata threshold to make a minimum-level gray-scale isodata image
        arr = np.shape(gray_scale)  
        for i in range(0,arr[0]):
            for j in range(0,arr[1]):
                if gray_scale[i,j] >= thresh:
                    gray_scale[i,j] = 255
                else: 
                    gray_scale[i,j] = 0
                   
                    
        ##save the results of isodata threshold on the minimum-level gray-scale image
        plt.savefig(result_addrs + file + '_(3)isodata.jpg') 
       
       
        
        ##binary morphological filters on minimum-level gray-scale isodata image(opening, closing)
        def plot_comparison(original, filtered, filter_name):
        
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                           sharey=True)
            ax1.imshow(original, cmap=plt.cm.gray)
            ax1.set_title('original')
            ax1.axis('off')
            ax2.imshow(filtered, cmap=plt.cm.gray)
            ax2.set_title(filter_name)
            ax2.axis('off') 
        
        ##opening filter on minimum-level gray-scale isodata image    
        selem = disk(6) 
        opened = opening(gray_scale, selem)
        
        
        ##dilation filter on minimum-level gray-scale isodata image    
        selem = disk(6)
        dilated = dilation(opened, selem)
 
 
        ##closing on minimum-level gray-scale isodata image
        selem = disk(6)       
        closed = closing(dilated, selem)
        
        
        ###dilation on minimum-level gray-scale isodata image
        #selem = disk(6)  
        #dilated = dilation(closed, selem)    
        
        
        #eroded = erosion(orig_phantom, selem)
        #plot_comparison(orig_phantom, eroded, 'erosion')       
             
        
        ##save the minimum-level gray-scale isodata edited image  
        gray_scale = np.copy(closed)
        cv2.imwrite(result_addrs + file + '_(4)edited_isodata.jpg', gray_scale) 
         
         
               
        ##do canny edge detection on the minimum-level gray-scale isodata edited image
        edges = cv2.Canny(gray_scale,100,200)
        plt.subplot(121),plt.imshow(gray_scale,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(edges,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
               
        
        ##save the image afted applying canny edge detection on the minimum-level gray-scale isodata edited image
        gray_scale = np.copy(edges)
        cv2.imwrite(result_addrs + file + '_(5)edges.jpg', gray_scale)
        
        
        
        ##remove small objects
        rem_sm_obj = remove_small_objects(gray_scale,  0.01 * levelDimension[img_th][0] * 0.01 * levelDimension[img_th][1], connectivity=2)
        
        for i in range(0, levelDimension[img_th][1]):
            for j in range(0, levelDimension[img_th][0]):
                if rem_sm_obj[i][j] == False:
                    gray_scale[i][j] = 0
                else:
                    gray_scale[i][j] = 255
                    
        gray_scale = np.copy(rem_sm_obj)
        
        
        
        ##save the image after removing the small objects 
        cv2.imwrite(result_addrs + file + '_(51)rem_sm_obj.jpg', gray_scale)         
        
        ##calculate the number of contours on the       
        ret,thresh = cv2.threshold(gray_scale,127,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)                        
        
        

        ##omit small contours on the minimum-level gray-scale isodata gray_scale image
        for i in range(0,len(contours)):
            mini = max(levelDimension[img_th][0], levelDimension[img_th][1])
            maxi = 0
            
            for j in range(0,len(contours[i])):
                if contours[i][j][0][0] > maxi:
                    maxi = contours[i][j][0][0]
                if contours[i][j][0][0] < mini:
                    mini = contours[i][j][0][0]
                    
            numH = maxi-mini+1 
            
            
            mini = max(levelDimension[img_th][0], levelDimension[img_th][1])
            maxi = 0
    
            for j in range(0,len(contours[i])):
                if contours[i][j][0][1] > maxi:
                    maxi = contours[i][j][0][1]
                if contours[i][j][0][1] < mini:
                    mini = contours[i][j][0][1]
                    
            numW = maxi-mini+1  
     
     
            tr = levelDimension[img_th][0] * levelDimension[img_th][1]
            if numH*numW < 0.001 * tr:
                for j in range(0,len(contours[i])):
                    gray_scale[contours[i][j][0][1]][contours[i][j][0][0]] = 0                    
            if numH*numW < 0.01 * tr:
                if len(contours[i]) < 0.001:
                    for j in range(0,len(contours[i])):
                        gray_scale[contours[i][j][0][1]][contours[i][j][0][0]] = 0 
                        
        ##save the minimum-level gray-scale isodata edited image after omitting the small contours    
        cv2.imwrite(result_addrs + file + '_(6)edited_edges.jpg', gray_scale)       
        
        
        
        ##trace the minimum-level gray-scale isodata edited image after omitting the small contours to find the object zones (sweep the columns)
        sz = np.shape(gray_scale)
        objct = []
        num = -1
        cnt_pre = [0, 0, 0]
        cnt = 0

        for j in range(0, sz[1]):
            tmp = cnt_pre
            cnt_pre = [tmp[1], tmp[2], cnt] #
            cnt = 0
            for i in range(0,sz[0]):
                if gray_scale[i][j] == 255:
                    cnt = 1
                    continue
                
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
                      
                        
        ##trace the minimum-level gray-scale isodata edited image after omitting the small contours to find the object zones (sweep the rows)
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
                        cnt = cnt+1
                if cnt > 0:
                    if cnt_pre[0] + cnt_pre[1] + cnt_pre[2] == 0:                        
                        objct_list.append([i, i, objct[n][0], objct[n][1]]) 
                        objct_num = objct_num+1
                        
                    else:
                        nc = objct_list.pop()
                        objct_list.append([nc[0], i, nc[2], nc[3]])
                                          
                else:
                    if cnt_pre[0] == 1 and cnt_pre[1]  == 0 and cnt_pre[2] == 0:                           
                        if (objct_list[objct_num][1] - objct_list[objct_num][0]) < (0.05 * sz[0]):
                            for k in range(objct[n][0], objct[n][1]+1):
                                for t in range(objct_list[objct_num][0], objct_list[objct_num][1]+1):
                                    gray_scale[t][k] = 0 
                                
                            del objct_list[objct_num]
                            objct_num = objct_num-1
                            
                        else:
                            find = False
                            for m in range(objct_list[objct_num][2], objct_list[objct_num][3]+1):
                                strt = objct_list[objct_num][2]                                
                                for n in range(objct_list[objct_num][0], objct_list[objct_num][1]+1):
                                    if gray_scale[n][m] == 255:
                                        strt = m
                                        find = True
                                        break
                                        
                                if find == True:
                                    break
                            for m in range(objct_list[objct_num][3], str-1, -1):
                                nd = objct_list[objct_num][3]                                
                                for n in range(objct_list[objct_num][0], objct_list[objct_num][1]+1):
                                    if gray_scale[n][m] == 255:
                                        nd = m
                                        break
                                        
                            if (nd - strt) < (0.05 * sz[1]):
                                for k in range(objct[n][0], objct[n][1]+1):
                                    for t in range(objct_list[strt][0], objct_list[nd][1]+1):
                                        gray_scale[t][k] = 0 
                                    
                                del objct_list[objct_num]
                                objct_num = objct_num-1      
                                  
                                                                    

        objct_num = objct_num+1
        ##calculate minimum size of object width/higth (need to compare with patch_size)
        list_size = len(objct_list)
        if list_size < 1:
            #print file            
            continue

        
        
        ###print the number of detected objects
        #print "number of detected objects are: ", objct_num
        #if objct_num < 2 or objct_num > 5:
            #print "for the file ", file, " number of detected ojects is strange, maybe you should check!"


        ## save the minimum-level gray-scale isodata edited image after omitting the small contours and detecting the objects    
        cv2.imwrite(result_addrs + file + '_(7)object_detection.jpg', gray_scale)
         
              
        
        ##drow a convex hall around each object in the minimum-level gray-scale isodata edited image after omitting the small contours and detecting the objects
        for k in range(0,objct_num):            
            mat = np.zeros(shape=(levelDimension[img_th][1],levelDimension[img_th][0]))
            
            for i in range(objct_list[k][0], objct_list[k][1]+1):
                for j in range(objct_list[k][2], objct_list[k][3]+1):
                    mat[i][j] = gray_scale[i][j]
                    
            chull = convex_hull_image(mat)
            
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            ax = axes.ravel()
            
            ax[0].set_title('Original picture')
            ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
            ax[0].set_axis_off()
            
            ax[1].set_title('Transformed picture')
            ax[1].imshow(chull, cmap=plt.cm.gray, interpolation='nearest')
            ax[1].set_axis_off()
            
            plt.tight_layout()
                
                                
            for i in range(objct_list[k][0], objct_list[k][1]+1):
                for j in range(objct_list[k][2], objct_list[k][3]+1):
                    if chull[i][j] == False:
                        gray_scale[i][j] = 0
                    else:
                        gray_scale[i][j] = 255
                                                        
            ##save the objects with a convex hull around
            #plt.savefig(result_addrs + file + '_covex_hull'+ str(k) +'.jpg')  
        
        ##save the objects on the smallest scale of the image    
        cv2.imwrite(result_addrs + file + '_(8)convex_hull.jpg', gray_scale)  
        
        
            
        ##resize the image
        #print "There are different magnification levels,"
        #for i in range(0, img_th): 
            #print "level ", i, ":", levelDimension[i]
       
        
       
        ###convertion from text to ASCII codes, to accept the correct magnification layer
        ##inpt = raw_input("Please choose the desired manginifasion layer: ") 
        #inpt = str(random.randint(0, img_th-1))
        #sm = 0;
        #for ch in inpt:
            #sm = sm + ord(ch)
            
        #ln = 0
        #for ch in str(img_th-1):
            #ln = ln + ord(ch)
            
        #while sm < 48 or sm > ln:        
            ##inpt = raw_input("Please insert the desired manginifasion layer: ") 
            #inpt = str(random.randint(0, img_th-1))
            #sm = 0;
            #for ch in inpt:
                #sm = sm + ord(ch)      
  
        #resize_ = levelDimension[int(inpt)][0]/levelDimension[levelCount-1][0]
        
        #if type(resize_) != int:
            #print "resize_ is not integer!"
            #continue
        
        #resized_matrix = np.repeat(np.repeat(gray_scale, resize_, axis=0), resize_, axis=1)
     
            
        
        #min_pos_patch_sz = resize_ * (objct_list[0][1] - objct_list[0][0]+1)
        #if min_pos_patch_sz > resize_ * (objct_list[0][3] - objct_list[0][2]+1):
            #min_pos_patch_sz = resize_ * (objct_list[0][3] - objct_list[0][2]+1)
        #for i in range(1, list_size):
            #if min_pos_patch_sz > resize_ * (objct_list[0][1] - objct_list[0][0]+1):
                #min_pos_patch_sz = resize_ * (objct_list[0][1] - objct_list[0][0]+1)
            #if min_pos_patch_sz > resize_ * (objct_list[0][3] - objct_list[0][2]+1):
                #min_pos_patch_sz = resize_ * (objct_list[0][3] - objct_list[0][2]+1)            
            
               
        
        ####define a patch size
        ##inpt = raw_input("Please insert the desired patch size:")
        #if min_pos_patch_sz > 10:
            #mini = 10
        #else:
            #mini = 2
        #inpt = str(random.randint(mini, min_pos_patch_sz))
        
        #sm = 0
        #for ch in inpt:
            #sm = sm + ord(ch)
            
        #ln1 = 0        
        #for ch in str(mini):
            #ln1 = ln1 + ord(ch)
        
        #ln2 = 0        
        #for ch in str(100):
            #ln2 = ln2 + ord(ch)
            
        #while sm < ln1 or sm > ln2:        
            ##inpt = raw_input("Please insert the desired patch size:") 
            #inpt = str(random.randint(10, 100))
            #sm = 0
            #for ch in inpt:
                #sm = sm + ord(ch) 
                
        #patch_size = int(inpt)
        
        
          
        ####overlap size for the patches
        ##inpt = raw_input("Please insert the horisental_overlaping percentage:(between 0 and 99) ")
        #inpt = str(random.randint(0, 99))
        
        #sm = 0
        #for ch in inpt:
            #sm = sm + ord(ch)
            
        #ln1 = ord(str(0))
        
        #ln2 = 0        
        #for ch in str(99):
            #ln2 = ln2 + ord(ch)
            
        #while sm < ln1 or sm > ln2:        
            ##inpt = raw_input("Please insert the horisental_overlaping percentage:(between 0 and 99) ") 
            #inpt = str(random.randint(0, int(99 * patch_size / 100)))
            #sm = 0;
            #for ch in inpt:
                #sm = sm + ord(ch)                 
        
        #hor_ovlap = int(int(inpt)*patch_size/100)
        
        #if hor_ovlap == patch_size:
            #hor_ovlap = hor_ovlap-1
            
           
        
        
        ##inpt = raw_input("Please insert the vertical_overlaping percentage:(between 0 and 99) ")
        #inpt = str(random.randint(0, int(int(99 * patch_size / 100) * patch_size / 100)))
        
        #sm = 0
        #for ch in inpt:
            #sm = sm + ord(ch)
            
        #ln1 = ord(str(0))
        
        #ln2 = 0        
        #for ch in str(99):
            #ln2 = ln2 + ord(ch)
            
        #while sm < ln1 or sm > ln2:        
            ##inpt = raw_input("Please insert the vertical_overlaping percentage:(between 0 and 99) ") 
            #inpt = str(random.randint(0, 99))
            #sm = 0;
            #for ch in inpt:
                #sm = sm + ord(ch)         
        
        #ver_ovlap = int(int(inpt)*patch_size/100)        
           
        #if ver_ovlap == patch_size:
            #ver_ovlap = ver_ovlap-1
            
            
        
        ###choose the minimum coverage of the convex hull with the patches 
        #per_ = 80 / 100
        
        
        ###trace objects of the image based on the patches
        #patch_list = list()
        #for n in range(0, objct_num): 
            #p = 0
            
            #for i in range(0,int((objct_list[n][1] - objct_list[n][0]) * resize_ - ver_ovlap) / (patch_size - ver_ovlap)):
                #for j in range(0,int((objct_list[n][3] - objct_list[n][2]) * resize_ - hor_ovlap) / (patch_size - hor_ovlap)):
                    #w = 0
                    #tmp = [[-1 for x in range(0,patch_size)] for y in range(0,patch_size)]
                    
                    #for ip in range(i * (patch_size - ver_ovlap) + (objct_list[n][0] * resize_), i * (patch_size - ver_ovlap) + (objct_list[n][0] * resize_) + patch_size):
                        #for jp in range(j * (patch_size - hor_ovlap) + objct_list[n][0], j * (patch_size - hor_ovlap) + objct_list[n][0] + patch_size):
                            #tmp[ip % patch_size][jp % patch_size] = resized_matrix[ip][jp]
                            #if resized_matrix[ip][jp] == 0:
                                #w = w+1
                                
                    
                    #if w > per_ * patch_size:                         
                        #patch_list.append([n, p, ip, jp])
                        #p = p+1
        
            
        ###save the patch list to a file
        #file_Name = result_addrs + file + "_patch_list"
        #fileObject = open(file_Name,'wb') 
        #pickle.dump(patch_list,fileObject)   
        #fileObject.close()
  
        ###save image information to the file
        #img_info.append([file, levelCount, levelDimension, objct_num, objct_list, resize_, patch_size, ver_ovlap, hor_ovlap])
        #plt.close('all')
        
        
        
###save the information for the the images in a file
#file_Name = result_addrs + "Images_INFO"
#fileObject = open(file_Name,'wb') 
#pickle.dump(img_info,fileObject)   
#fileObject.close()