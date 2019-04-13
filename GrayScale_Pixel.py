##This program extracts information related to the gray scale images.
##calculates the percentage of the appearance of each colour (0-255) 
import os
import openslide
import cv2
import numpy as np
from skimage.filters import threshold_isodata
from skimage.morphology import opening, closing, disk, dilation, convex_hull_image

img_addrs = "/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/"
result_addrs = "/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/Results/"

img_info = list()
                
for file in sorted(os.listdir(img_addrs)):
    if file.endswith(".ndpi"):       
    
        file_name = file.replace('.ndpi','')
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
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        
        pixel = np.zeros(shape=(1,256))
        for i in range(0,arr[0]):
            for j in range(0,arr[1]):
                pixel[0][gray_scale[i][j]] += 1
        for i in range(0,256):
            pixel[0][i] = pixel[0][i]/arr[0]*arr[1]
            
        img_info.append([file, pixel, iso_thresh]) 
        
adr = result_addrs + "pixel" 
np.save(adr, np.asanyarray(img_info))