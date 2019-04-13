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
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        iso_thresh = threshold_isodata(gray_scale)
        arr = np.shape(gray_scale)
        pixel = np.zeros(shape=(1,256))
        for i in range(0,arr[0]):
            for j in range(0,arr[1]):
                pixel[0][gray_scale[i][j]] += 1
      
        for i in range(0,256):
            pixel[0][i] = pixel[0][i]/arr[0]*arr[1]
            
        mn = 0
        for i in range(0,iso_thresh):
            mn += pixel[0][i]
        mx = 0
        for i in range(iso_thresh,255):
            mx += pixel[0][i]    
            
        sm = mn+mx
        img_info.append([file, iso_thresh, mn/sm, mx/sm], pixel) 
        
np.save(result_addrs + "pixel", np.asarray(img_info,dtype= object))
