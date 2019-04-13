# import the necessary packages
from matplotlib import pyplot as plt
import numpy as np
import argparse
import cv2
from PIL import Image

import openslide
import os
import matplotlib.image as mpimg


image = cv2.imread('/home/seirana/Documents/coins.jpg')

# convert the image to grayscale and create a histogram
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)




cv2.imshow("gray", gray)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.show()


