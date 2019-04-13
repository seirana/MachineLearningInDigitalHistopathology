import matplotlib.pyplot as plt 
import numpy as np
import glob
import os, sys
fpath ="path_Of_my_final_Big_File"
npyfilespath ="path_of_my_numpy_files"   
os.chdir(npyfilespath)
npfiles= glob.glob("*.npy")
npfiles.sort()
all_arrays = []
for i, npfile in enumerate(npfiles):
    all_arrays.append(np.load(os.path.join(npyfilespath, npfile)))
np.save(fpath, np.concatenate(all_arrays))