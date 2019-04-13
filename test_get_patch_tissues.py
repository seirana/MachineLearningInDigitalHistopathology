'''
    Extract patches from slides
'''
import numpy as np
import random
from extract_patches import read_patch

#def get_patch_tissuepeices(slide_number, image_resolusion, convexhull_resolusion, convex_hull, tissue_list):
    
slide = 1
patch_slide = 100
image_resolusion = 0
convexhull_resolusion = 5
convex_hull_magin = list()
tissue = list()
tissue_count = 4
resolution = 0
base_level = 5
mul = 2^(base_level-resolution-base_level)
tissue_space = np.zeros(tissue_count)
whole_space = 0
patch_list = list()

for i in range(0, tissue_count):
    tissue_space[i] += ((tissue['lowerbound'] - tissue['upperbound']+1) * (tissue['rightbound'] - tissue['leftbound']+1))        
    whole_space += tissue_space[i]

sm = 0  
patch_tissue = list() 
for i in range(0,tissue_count-1):
    patch_tissue[i] = np.floor(patch_slide * tissue_space[i] / whole_space)
    sm += patch_tissue[i]
patch_tissue[tissue_count-1] = patch_slide - sm

st=list()
nd = list()  

for i in range(0,tissue_count):
    ##multiply be 2^5 
    w = tissue['rightbound'] - tissue['leftbound']+1
    l = np.floor(w/10)
    space = np.zeros(10)
    total_space = 0
    for j in range(0,10):
        st[j] = l*j+tissue['leftbound']
        nd[j] = st[j]+l-1
        if j == 9:
            nd[j] = tissue['rightbound']
                
        for k in range(st[j],nd[j]):
            space[i] += tissue[k]['lowerbound']-tissue[k]['upperbound']+1
            total_space += space[i]
            
    space_list_length = 0        
    for j in range(0,10):
        space[k] /= total_space
         
        if 0 < space[j] and space[j] <= 0.1:
            w = 1        
        if 0.1 < space[j] and space[j] <= 0.2:
            w = 2
        if 0.2 < space[j] and space[j] <= 0.3:
            w = 3
        if 0.3 < space[j] and space[j] <= 0.4:
            w = 4
        if 0.4 < space[j] and space[j] <= 0.5:
            w = 5
        if 0.5 < space[j] and space[j] <= 0.6:
            w = 6
        if 0.6 < space[j] and space[j] <= 0.7:
            w = 7
        if 0.7 < space[j] and space[j] <= 0.8:
            w = 8       
        if 0.8 < space[j] and space[j] <= 0.9:
            w = 9
        if 0.9 < space[j] and space[j] <= 1:
            w = 10
 
        space[j]  = w
        space_list_length += space[j]
    
    space_list_length = int(space_list_length)
    
    for j in range(0,10):
        for k in range(0,):
        r_w=random.randint(st[j]*mul, (nd[j]+1)*mul-1)
        r_h=random.randint(tissue[r_w]['upperbound']*mul,(tissue[r_w]['lowerbound']+1)*mul-1)
        patch_list.append(read_patch(r_w,r_h))

    
            