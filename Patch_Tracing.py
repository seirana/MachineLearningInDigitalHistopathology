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


resize_ = 5
objct_list = [10, 41, 50, 71]
ver_ovlap = 5
patch_size = 23
hor_ovlap = 7
resized_matrix = np.zeros(shape=(55* resize_, 100*resize_))
  
     
for i in range(0,int(((objct_list[1] - objct_list[0]) * resize_ - ver_ovlap) / (patch_size - ver_ovlap))):
    for j in range(0,int(((objct_list[3] - objct_list[2]) * resize_ - hor_ovlap) / (patch_size - hor_ovlap))):       
        for ip in range(i * (patch_size - ver_ovlap) + (objct_list[0] * resize_), i * (patch_size - ver_ovlap) + (objct_list[0] * resize_) + patch_size):
            for jp in range(j * (patch_size - hor_ovlap) + objct_list[2], j * (patch_size - hor_ovlap) + objct_list[2] + patch_size):
                resized_matrix[ip][jp] = resized_matrix[ip][jp]+1
                    
for i in range(0,int(((objct_list[1] - objct_list[0]) * resize_ - ver_ovlap) / (patch_size - ver_ovlap))):
    for j in range(0,int(((objct_list[3] - objct_list[2]) * resize_ - hor_ovlap) / (patch_size - hor_ovlap))): 
        resized_matrix[i][j] = str(resized_matrix[i][j])
        
np.savetxt("/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/Results/resized_matrix.csv", resized_matrix, delimiter=",")
