# Importing the modules
import openslide
from Random_Rotation_Mirroring import random_rot_mirr
import numpy as np


# read the open slide
def read_openslide(addrs, slide_ID, leftup_h, leftup_w, patch_size, tt):
    slide = openslide.OpenSlide(addrs + slide_ID+'.ndpi')

    # read the object in the max. magnification level
    heigth_ = patch_size
    width_ = patch_size

    img = slide.read_region((leftup_w, leftup_h), 0, (width_, heigth_))
    img = np.array(img)
    patch = img[:, :, 0:3]  # remove the alpha channel
    # mirror and rotate the patch randomly
    patch = random_rot_mirr(patch, addrs, slide_ID, leftup_w, leftup_h, tt)
    
    return patch  # it must be a 3D matrix
