'''
This module receive a patch_ and returns onother matrix by randomly mirroring/rotationg the patch_ matrix 
Simply rotation 0, 90, 180 or 270 degree or rotation plus mirroring
'''
import numpy as np
import random

def random_rot_mirr(patch_):

    ##rotate a patch randomly based on a random number
    i = random.randint(0, 7)
    
    if i == 1:
        patch_ = np.rot90(patch_, k=1, axes=(0,1)) #90 ccw degree rotation
    if i == 2:
        patch_ = np.rot90(patch_, k=2, axes=(0,1)) #180 ccw degree rotation
    if i == 3:
        patch_ = np.rot90(patch_, k=3, axes=(0,1)) #270 ccw degree rotation 
    if i == 4:
        patch_ = np.flip(np.rot90(patch_, k=0), (0,1)) #0 ccw degree rotation with mirroring
    if i == 5:
        patch_ = np.flip(np.rot90(patch_, k=1), (0,1))#90 ccw degree rotation with mirroring
    if i == 6:
        patch_ = np.flip(np.rot90(patch_, k=2), (0,1)) #180 ccw degree rotation with mirroring
    if i == 7:
        patch_ = np.flip(np.rot90(patch_, k=3), (0,1)) #270 ccw degree rotation with mirroring
        
    return patch_