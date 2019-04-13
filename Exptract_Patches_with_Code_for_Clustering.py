"""
    read .ndpi files and extract patches from them and change them 
    to code and then send them to Cluster machine
"""
# Importing modules
import numpy as np
import openslide
import h5py


def read_openslide_get_patches(addrs, patch_sz, destination_Mag_lev):
    
    # read the object in the max. magnification level    
    result_addrs = "/home/seirana/Desktop/Workstation/casp3/patches_for_clustering/"
    img_lst = np.load(addrs + "_SlidesInfo_dic.npy")[()]
    heigth_ = patch_sz
    width_ = patch_sz 
       
    for slide in img_lst.item():
        patches = list()
        resize = slide.item().get(slide)['magnification_level'] - destination_Mag_lev
        margins = np.load(addrs + slide['slide_ID']+ "_margin.npy")[()]
        slide = openslide.OpenSlide(addrs + slide['slide_ID'] + '.ndpi')
        rs = resize 
        
        for tissue in (0,len(margins)):
            left = margins[0][0]
            right = margins[0][1]
            for left_w in range(left*rs, (right+1)*(rs+1)-1):
                for leftup_h in range(margins[left_w-left][0]*rs,margins[left_w-left][1]*(rs+1)-1):
                    img = slide.read_region((left_w, leftup_h), destination_Mag_lev, (width_, heigth_))
                    img = np.array(img)
                    patch = img[:, :, 0:3]  # remove the alpha channel
                    patches.append(patch)
                    
        with h5py.File(result_addrs+slide['slide_ID']+'patches.h5', 'w') as hf:
            hf.create_dataset('dataset', data=patches)                
    return 
