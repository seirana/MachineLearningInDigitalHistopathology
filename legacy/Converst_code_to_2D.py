# Importing the modules
import os
import numpy as np

os.system('clear')
addrs = "/home/seirana/Desktop/Workstation/casp3/"
result_addrs = "/home/seirana/Desktop/Workstation/casp3/code/"
img_lst = np.load(addrs + "_SlidesInfo_dic.npy")[()]
patch_size = 256
