"""
    This function procudes a random sampling list which means that we split batch size to randomly selected sections.
    to select uniform number of patches from each slide, we give weights to slides based on their tissue space    
"""

import numpy as np
import random


def make_random_list(addrs, batch_sz, tt):
    tissues_slides = np.load(addrs+'tissues_slides_space_percentage.npy')  # is a list
    train_test = np.load(addrs+'train_test_list.npy')[()]  # is a list
    slides = len(tissues_slides)  # number of slides
    weight_list_length = 0  # if slide one has weight = 3 and slide two has weight 4 then the weight_list_length = 7
    weight = np.zeros(slides)
    for i in range(0, slides):  # for all slides do
        if train_test[i] == tt:  # the slide selected for train set
            for j in range(0, len(tissues_slides[i])):
                weight[i] += tissues_slides[i][j]
                
            if 0 <= weight[i] <= 0.1:
                w = 1        
            if 0.1 < weight[i] <= 0.2:
                w = 2
            if 0.2 < weight[i] <= 0.3:
                w = 3
            if 0.3 < weight[i] <= 0.4:
                w = 4
            if 0.4 < weight[i] <= 0.5:
                w = 5
            if 0.5 < weight[i] <= 0.6:
                w = 6
            if 0.6 < weight[i] <= 0.7:
                w = 7
            if 0.7 < weight[i] <= 0.8:
                w = 8       
            if 0.8 < weight[i] <= 0.9:
                w = 9
            if 0.9 < weight[i] <= 1:
                w = 10
         
            weight[i] = w
            weight_list_length += weight[i]
    weight_list_length = int(weight_list_length)

    if tt == 'train':
        if batch_sz % weight_list_length == 0:
            train_list = np.zeros(weight_list_length)
        else: 
            train_list = np.load(addrs+'train_ranselected_weightbased_list.npy')
            
        while True:
            r = random.randint(0, weight_list_length -1)
            if train_list[r] == 0:
                train_list[r] = 1
                break
                
        while r - weight[i] > 0:
            r -= weight[i]
            i += 1
        random_selected_slide = i
        
    if tt == 'test':
        if batch_sz % weight_list_length == 0:
            test_list = np.zeros(weight_list_length)
        else: 
            test_list = np.load(addrs+'test_ranselected_weightbased_list.npy')
            
        while True:
            r = random.randint(0, weight_list_length -1)
            if test_list[r] == 0:
                test_list[r] = 1
                break
                
        while r - weight[i] > 0:
            r -= weight[i]
            i += 1
        random_selected_slide = i
    
    np.save(addrs+'train_ranselected_weightbased_list.npy', train_list)
    np.save(addrs+'test_ranselected_weightbased_list.npy', test_list)          
    return random_selected_slide
