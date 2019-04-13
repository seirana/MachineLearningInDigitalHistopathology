##This module reads .ndpi files from the source folder and save the attributes of the images as a .csv file in the destination folder
import os
import openslide
import numpy as np

##open the .ndpi file and find the level_count  and level dimmensions and other attributes of the image (if known)     
def slide_properties(img_addrs, result_addrs):    
    img_pro = list()
    img_pro.append(["File name", "%", "Level count", "%", "Level dimension", "%", "Quickhash", "%", "Objective power", "%" \
                           , "Vendor", "%", "Bounds color", "%", "comment", "%", "MAPP X", "%", "MPP Y", "%", "Bounds X", "%" \
                           , "Bounds Y", "%", "Bounds width", "%", "Bounds height"])
    
    for file in sorted(os.listdir(img_addrs)):
        if file.endswith(".ndpi"): 
            file_name = file.replace('.ndpi','') 
            slide = openslide.OpenSlide(img_addrs + file)
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
            
            img_pro.append([file_name, "%", levelCount, "%", levelDimension, "%", w0, "%", w1, "%", w2, "%", w3, "%", w4, "%" \
                                   , w5, "%", w6, "%", w7, "%", w8, "%", w9, "%", w10, "%", w11])        
            

    np.savetxt(result_addrs + "Image_Information.csv", img_pro, fmt='%s') 
    return