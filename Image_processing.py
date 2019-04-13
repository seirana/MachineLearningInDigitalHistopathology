import os
os.system('clear')

import numpy as np

from Image_Properties import slide_properties 
from patch_features import image_info
from Preprocesing import Preprocessing

img_addrs = "/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/"
result_addrs = "/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/Results/"

slide_properties(img_addrs, result_addrs)
img_size, patch_size, ver_ovlap, hor_ovlap, per = image_info()

information = list()

for file in sorted(os.listdir(img_addrs)):
    if file.endswith(".ndpi"):
        information = Preprocessing(img_addrs, result_addrs, file, information)
    else:
        continue
    
np.save(result_addrs + "_info_gathering", np.asarray(information,dtype= object))