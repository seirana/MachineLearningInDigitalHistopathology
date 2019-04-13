##This module receive a patch_ and returns onother matrix by randomly mirroring/rotationg the patch_ matrix 
##Simply rotation 0, 90, 180 or 270 degree or rotation plus mirroring
import numpy as np
import random


def random_rot_mirr(patch_):

    ##rotate a patch randomly based on a random number
    i = random.randint(0, 7)
    
    if i == 0:
        rand_rot_mirr = np.rot90(patch_, k=0) #0 degree roration 
    if i == 1:
        rand_rot_mirr = np.rot90(patch_, k=1) #90 cw degree rotation
    if i == 2:
        rand_rot_mirr = np.rot90(patch_, k=2) #180 cw degree rotation
    if i == 3:
        rand_rot_mirr = np.rot90(patch_, k=3) #270 cw degree rotation 
    if i == 4:
        rand_rot_mirr = np.fliplr(np.rot90(patch_, k=0)) #0 cw degree rotation with mirroring
    if i == 5:
        rand_rot_mirr = np.fliplr(np.rot90(patch_, k=1)) #90 cw degree rotation with mirroring
    if i == 6:
        rand_rot_mirr = np.fliplr(np.rot90(patch_, k=2)) #180 cw degree rotation with mirroring
    if i == 7:
        rand_rot_mirr = np.fliplr(np.rot90(patch_, k=3)) #270 cw degree rotation with mirroring
        
    patch_ = rand_rot_mirr
    
    return patch_