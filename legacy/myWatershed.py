#import numpy as np
#import cv2 as cv
#from matplotlib import pyplot as plt
#img = cv.imread('/home/seirana/Documents/coins.jpg')
#gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

## noise removal
#kernel = np.ones((3,3),np.uint8)
#opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
## sure background area
#sure_bg = cv.dilate(opening,kernel,iterations=3)
## Finding sure foreground area
#dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
#ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
## Finding unknown region
#sure_fg = np.uint8(sure_fg)
#unknown = cv.subtract(sure_bg,sure_fg)

## Marker labelling
#ret, markers = cv.connectedComponents(sure_fg)
## Add one to all labels so that sure background is not 0, but 1
#markers = markers+1
## Now, mark the region of unknown with zero
#markers[unknown==255] = 0

#markers = cv.watershed(img,markers)
#img[markers == -1] = [255,0,0]
##################################################################################################
#import gc
#gc.collect()  

#import numpy as np
#import matplotlib.pyplot as plt
#from scipy import ndimage as ndi

#from skimage.morphology import watershed
#from skimage.feature import peak_local_max

## Generate an initial image with two overlapping circles
#x, y = np.indices((80, 80))
#x1, y1, x2, y2 = 28, 28, 44, 52
#r1, r2 = 16, 20
#mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
#mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
#image = np.logical_or(mask_circle1, mask_circle2)

## Now we want to separate the two objects in image
## Generate the markers as local maxima of the distance to the background
#distance = ndi.distance_transform_edt(image)
#local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                            #labels=image)
#markers = ndi.label(local_maxi)[0]
#labels = watershed(-distance, markers, mask=image)

#fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
#ax = axes.ravel()

#ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
#ax[0].set_title('Overlapping objects')
#ax[1].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
#ax[1].set_title('Distances')
#ax[2].imshow(labels, cmap=plt.cm.nipy_spectral, interpolation='nearest')
#ax[2].set_title('Separated objects')

#for a in ax:
    #a.set_axis_off()

#fig.tight_layout()
#plt.show()

################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.morphology import watershed
from skimage.feature import peak_local_max


# Generate an initial image with two overlapping circles
x, y = np.indices((80, 80))
x1, y1, x2, y2 = 28, 28, 44, 52
r1, r2 = 16, 20
mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
image = np.logical_or(mask_circle1, mask_circle2)

# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance to the background
distance = ndi.distance_transform_edt(image)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                            labels=image)
markers = ndi.label(local_maxi)[0]
labels = watershed(-distance, markers, mask=image)

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax[0].set_title('Overlapping objects')
ax[1].imshow(-distance, cmap=plt.cm.gray, interpolation='nearest')
ax[1].set_title('Distances')
ax[2].imshow(labels, cmap=plt.cm.nipy_spectral, interpolation='nearest')
ax[2].set_title('Separated objects')

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()