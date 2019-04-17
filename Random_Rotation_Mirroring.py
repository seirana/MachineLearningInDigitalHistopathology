"""
This module receive a patch_ and returns onother matrix by randomly mirroring/rotationg the patch_ matrix 
Simply rotation 0, 90, 180 or 270 degree or rotation plus mirroring
"""
import numpy as np
import random


def random_rot_mirr(patch, addrs, leftup_w, leftup_h, tt, slide_th):
    # rotate a patch randomly based on a random number
    i = random.randint(0, 7)
    if i == 0:
        rot = 'none' 
    if i == 1:
        patch = np.rot90(patch, k=1, axes=(0, 1))  # 90 ccw degree rotation
        rot = '90degree_rotation'
    if i == 2:
        patch = np.rot90(patch, k=2, axes=(0, 1))  # 180 ccw degree rotation
        rot = '180degree_rotation'
    if i == 3:
        patch = np.rot90(patch, k=3, axes=(0, 1))  # 270 ccw degree rotation
        rot = '270degree_rotation'
    if i == 4:
        patch = np.flip(np.rot90(patch, k=0), (0, 1))  # 0 ccw degree rotation with mirroring
        rot = 'mirroring'
    if i == 5:
        patch = np.flip(np.rot90(patch, k=1), (0, 1))  # 90 ccw degree rotation with mirroring
        rot = '900degree_rotation+mirroring'
    if i == 6:
        patch = np.flip(np.rot90(patch, k=2), (0, 1))  # 180 ccw degree rotation with mirroring
        rot = '270degree_rotation+mirriring'
    if i == 7:
        patch = np.flip(np.rot90(patch, k=3), (0, 1))  # 270 ccw degree rotation with mirroring
        rot = '270degree_rotation+mirriring'

    patch_list = np.load(addrs+'patch_train_test_list.npy')[()]
    patch_list[len(patch_list)+1] = {'slide_number': slide_th, 'height': leftup_h, 'weidth': leftup_w, 'ritation': rot, 'test_train': tt}
    np.save(addrs+'patch_train_test_list.npy', patch_list)
    return patch
