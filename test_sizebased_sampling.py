'''
Test random patch size selection for slides = s, batch size = b & epochs = e     
'''
import os
os.system('clear')

import numpy as np
import random

import time
start = time.time()

import psutil
print(psutil.virtual_memory())

#def make_random_list(slides, batch_size, batch_num, epochs, resolution_level):
        
slides = 8
batch_size = 200
batch_num = 1000
epochs = 1000
tries = list()

weight = np.zeros(slides)
resolution_level = 5
#glassslide_x = np.floor((76*10^6)/((2^resolution_level)*226))
#glassslide_y = np.floor((26*10^6)/((2^resolution_level)*226))
glassslide_space = 100 #glassslide_x * glassslide_y

tissue_peices = [[11,23],[55],[21,22,15],[5],[13,8,3,20],[31,32],[25,26],[15,6,17,18]] #list()
weight_list_length = 0

for i in range(0,slides):
    weight[i] = 0
    for j in range(0,len(tissue_peices[i])):
        weight[i] += tissue_peices[i][j]  
      
    weight[i]/= glassslide_space
    
    if 0 < weight[i] and weight[i] <= 0.1:
        w = 1        
    if 0.1 < weight[i] and weight[i] <= 0.2:
        w = 2
    if 0.2 < weight[i] and weight[i] <= 0.3:
        w = 3
    if 0.3 < weight[i] and weight[i] <= 0.4:
        w = 4
    if 0.4 < weight[i] and weight[i] <= 0.5:
        w = 5
    if 0.5 < weight[i] and weight[i] <= 0.6:
        w = 6
    if 0.6 < weight[i] and weight[i] <= 0.7:
        w = 7
    if 0.7 < weight[i] and weight[i] <= 0.8:
        w = 8       
    if 0.8 < weight[i] and weight[i] <= 0.9:
        w = 9
    if 0.9 < weight[i] and weight[i] <= 1:
        w = 10
 
    weight[i]  = w
    weight_list_length += weight[i]
   
weight_list_length = int(weight_list_length)

for i in range(0,epochs):
    for j in range(0,batch_num):
        lst = []
        while len(lst) < weight_list_length-1:
            r=random.randint(1,batch_size)
            if r not in lst: lst.append(r)
            
        lst = np.sort(lst)
        lst_s = np.zeros(weight_list_length)
        lst_s[0] = lst[0]
        for k in range(1,weight_list_length-1):
            lst_s[k] = lst[k]- lst[k-1]
        lst_s[weight_list_length-1] = batch_size - lst[weight_list_length-2]
        slide_weight = np.zeros(slides)
        cnt = 0
        for k in range(0,slides):
            slide_weight[k] = sum(lst_s[int(cnt):int(weight[k])+int(cnt)])
            cnt += weight[k]
    
    tries.append(slide_weight)
       
print(psutil.virtual_memory()) 
end = time.time()
print(end - start)
        
ave_per_slide = np.zeros(slides) 
for i in range(0,slides):
    for j in range(0,len(tries)):
        ave_per_slide[i]+= tries[j][i]
    
    ave_per_slide[i]/=len(tries) 
    ave_per_slide[i]= ave_per_slide[i]*weight_list_length/weight[i]

print(ave_per_slide)
       
np.save("/home/seirana/Documents/list", tries)
    #return
