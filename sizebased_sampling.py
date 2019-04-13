'''
    This function procudes a random sampling list which means that we split batch size to randomly selected sections.
    to select uniform number of patches from each slide, we give weights to slides based on their tissue space    
'''

import numpy as np
import random

def make_random_list(addrs, batch_size):

    tissues_slides = np.load(addrs+'tissues_slides_space_percentage.npy') #is a list
    train_test = np.load(addrs+'train_test_list.npy') #is a list
    slides = len(tissues_slides) #number of slides
    weight_list_length = 0 #if slide one has weight = 3 and slide two has weight 4 then the weight_list_length = 7
    weight = np.zeros(slides)
    for i in range(0,slides): # for all slides do
        if train_test[i] == 1: #the slide selected for train set
            for j in range(0,len(tissues_slides[i])):
                weight[i] += tissues_slides[i][j]
                
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
    for i in range(0,slides):
         if train_test[i] == 1: 
            for j in range(cnt,cnt+int(weight[i])):
                slide_weight[i] += lst_s[j]
            cnt += int(weight[i])   
            
    return slide_weight