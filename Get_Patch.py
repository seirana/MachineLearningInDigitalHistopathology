##This program reads the patches from the files and rebuilds the convex hulls 
import os
import numpy as np
from PIL import Image

img_addrs = "/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/Test/"

for file in sorted(os.listdir(img_addrs)):
    if file.endswith("_patch_list.npy"):    
        file_name = img_addrs + file        
        patch_list = np.load(img_addrs+file_name)
        #file_name = file.replace('list','info')        
        patch_info = np.load(img_addrs+file_name)
        print (np.shape(patch_info))
        print (patch_info[0])
        strt = 0
        print (patch_info[0][0][4])
        for i in range(0,patch_info[0][0][4]):
            resize = patch_info[i][0][6]
            obj_width = (patch_info[0][0][5][i][1]-patch_info[0][0][5][i][0])*resize
            obj_heigth = (patch_info[0][0][5][i][3]-patch_info[0][0][5][i][2])*resize            
            mat = np.zeros(shape=(obj_width,obj_heigth,3), dtype=np.uint8)
            for j in range(strt,len(patch_info)):
                if i == patch_info[j][1]:                    
                    s1 = patch_info[j][2]-patch_info[0][0][5][i][0]*resize
                    s2 = patch_info[j][3]-patch_info[0][0][5][i][2]*resize
                    mat[s1:s1+100,s2:s2+100,:]= patch_list[j,:,:,0:3]
                    pix = np.zeros(shape=(100,100,3), dtype=np.uint8)
                    pix[:,:,:]= patch_list[j,:,:,0:3]
                    img = Image.fromarray(pix, 'RGB')
                    img.save(img_addrs+'test-image-cover_pix'+str(j)+'.png') #converts each patch to an image
                else:
                    if j > patch_info[i][0][4]:
                        strt  = j
                        break
    else:
        continue       
        