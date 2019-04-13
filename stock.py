##  Read te RGB value of a given pixel
#from PIL import Image

#im = Image.open('dead_parrot.jpg') # Can be many different formats.
#pix = im.load()
#print im.size  # Get the width and hight of the image for iterating over
#print pix[100,100]  # Get the RGBA Value of the a pixel of an image
#pix[100,100] = value  # Set the RGBA Value of the image (tuple)
#im.save('alive_parrot.png')  # Save the modified pixels as .png
           
###--------------------------------------------------------------------------------- 

## read .ndpi file with openslide
#import openslide
import os

#from skimage import io as io
import scipy.misc as misc
import matplotlib.pyplot as plt

img = '/home/seirana/Documents/coins.jpg'
imgA = misc.imread(img)

#import matplotlib.image as mpimg
plt.imshow(imgA)

##Reading files from a directory (way 1)
#os.chdir("/home/seirana/Documents")
#for file in glob.glob("*.ndpi"):
    #slide = openslide.OpenSlide(file)
    #img = slide.get_thumbnail((1000,1000))
    #img.show()    
    #levelCount = slide.level_count
    #print "level_count:", levelCount
 
##Reading files from a directory (way 2)    
#for file in os.listdir("/home/seirana/Documents"):
    #if file.endswith(".ndpi"):
        #slide = openslide.OpenSlide(file)
        
                
        #levelCount = slide.level_count
        #print "level_count:", levelCount        
        
            
        #levelDimension = slide.level_dimensions
        #print " level_dimensions:", levelDimension
        
        #x = levelDimension[levelCount-1]             
        #img = slide.get_thumbnail(x)
        #img.show()   
        

#Reading files from a directory (way 3)
#for root, dirs, files in os.walk("/home/seirana/Documents"):
    #for file in files:
        #if file.endswith(".ndpi"):        
            #slide = openslide.OpenSlide(file)
            #img = slide.get_thumbnail((1000,1000))
            #img.show()    
            #levelCount = slide.level_count
            #print "level_count:", levelCount
            
###---------------------------------------------------------------------------------           

#import PIL
#import cv2
#import numpy
#import matplotlib
#import scipy
#import glob


##import mat file
#import scipy.io as spio
#mat = spio.loadmat('control_rgb_50.mat', squeeze_me=True)
#mat.items()

##Load MATLAB file.
#loadmat(file_name[, mdict, appendmat])
##Save a dictionary of names and arrays into a MATLAB-style .mat file.
#savemat(file_name, mdict[, appendmat, …])
##List variables inside a MATLAB file.
#whosmat(file_name[, appendmat])
            
###--------------------------------------------------------------------------------- 

##opening images in python
#import numpy as np
#from PIL import Image

#mg = read_region(location=(0, 0), level=0, size=slide.dimensions)

#import cv2
#import numpy as np
#from matplotlib import pyplot as plt

#img = cv2.imread('messi5.jpg',0)
#edges = cv2.Canny(img,100,200)

#plt.subplot(121),plt.imshow(img,cmap = 'gray')
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#plt.show()
           
###--------------------------------------------------------------------------------- 

#import numpy as np
#import cv2
#from matplotlib import pyplot as plt

#img = cv2.imread('water_coins.jpg')
#gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
## noise removal
#kernel = np.ones((3,3),np.uint8)
#opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
           
###--------------------------------------------------------------------------------- 