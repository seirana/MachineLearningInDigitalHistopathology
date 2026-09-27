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

##begin-the code
slides = 8
batch_size = 200
batch_num = 1000
epochs = 1000
tries = list()

for i in range(0,epochs):
    for j in range(0,batch_num):
        lst = np.zeros(slides)
        counter = -1
        b = batch_size
        while counter < slides-2 and b > 0:
            counter += 1
            lst[counter] = random.randint(0,b)
            b -= lst[counter]
            
        lst[len(lst)-1] = b
        random.shuffle(lst)
        
        tries.append(lst)
        
print(psutil.virtual_memory()) 
end = time.time()
print(end - start)
        
ave_per_slide = np.zeros(slides) 
for i in range(0,slides):
    for j in range(0,len(tries)):
        ave_per_slide[i]+= tries[j][i]
    
    ave_per_slide[i]/=len(tries)

print(ave_per_slide)
       
np.save("/home/seirana/Documents/list", tries)
##end-the code