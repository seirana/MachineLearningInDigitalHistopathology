import random
import numpy as np

patch_tissue = 20

up = 0
down = 7
left = 0
right =7

patch_size = 1

margins = [[0,7],[2,4],[2,4],[1,5],[1,5],[0,7],[1,5],[1,5],[2,4]]
fail = np.zeros(100000)
f = 0
mat = np.zeros(shape=(8,8))
generated_patches = np.zeros(100000)
for i in range(0,100000):
    while generated_patches[i] < patch_tissue:
    #make random numbers inside the rectangle that coverd the tissue
        r_w=random.randint(up, down-patch_size+1) #check if the random numbers are inside the convex hull
        r_h=random.randint(left,right-patch_size+1)   
        if r_h >= margins[r_w+1][0] and r_h <= margins[r_w+1][1]:
            generated_patches[i] += 1
            mat[r_h][r_w] += 1
        else:
            fail[i] += 1
            f += 1
                
print (mat, f/100000)                