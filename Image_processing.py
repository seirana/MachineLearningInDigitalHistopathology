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
pixel_info = list()

for file in sorted(os.listdir(img_addrs)):
    if file.endswith(".ndpi"):
        information, pixel_info = Preprocessing(img_addrs, result_addrs, file, img_size, patch_size, ver_ovlap, hor_ovlap, per, information, pixel_info)
        np.save(result_addrs + "_info_gathering", np.asarray(information,dtype= object))
        np.save(result_addrs + "GrayScale_pixels", np.asarray(pixel_info,dtype= object))
    else:
        continue