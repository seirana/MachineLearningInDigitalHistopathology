## Importing the modules
import openslide
from Random_Rotation_Mirroring import random_rot_mirr
import numpy as np

##read the open slide
def read_openslide(addrs, slide_ID, leftup_i, leftup_j, patch_size):
    slide = openslide.OpenSlide(addrs + slide_ID+'.ndpi')

    ##read the object in the max. magnification level
    heigth_ = patch_size
    width_ = patch_size

    img = slide.read_region((leftup_j, leftup_i), 0, (width_, heigth_))
    img = np.array(img)
    patch = img[:,:,0:3] #remove the alpha channel
    #mirror and rotate the patch randomly
    patch = random_rot_mirr(patch)

    return patch #it must be a 3D matrix   