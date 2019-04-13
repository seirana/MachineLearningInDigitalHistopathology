##comparing otsu global and local threshod on the minimum-level gray-scale image        
global_thresh = threshold_otsu(gray)
binary_global = gray > global_thresh

block_size = 35
local_thresh = threshold_local(gray, block_size, offset=10)
binary_local = gray > local_thresh

fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax = axes.ravel()
plt.gray()

ax[0].imshow(gray)
ax[0].set_title('Original')

ax[1].imshow(binary_global)
ax[1].set_title('Global thresholding')

ax[2].imshow(binary_local)
ax[2].set_title('Local thresholding')

for a in ax:
    a.axis('off')
    
plt.savefig('/home/seirana/Documents/Results/threshold_otsu_Local&Global.jpg')  



##save the minimum-level gray-scale otsu image           
cv2.imwrite('/home/seirana/Documents/Results/otsu'+ file +'.jpg', gray) 



##black_tophat on minimum-level gray-scale otsu image
selem = disk(1)
b_tophat = black_tophat(opened, selem)
plot_comparison(opened, b_tophat, 'black tophat') 


##erosion on minimum-level gray-scale otsu image
selem = disk(1) 
eroded = erosion(opened, selem)
plot_comparison(opened, eroded, 'erosion')        

##dilation on minimum-level gray-scale otsu image
selem = disk(1) 
dilated = dilation(eroded, selem)
plot_comparison(eroded, dilated, 'dilation') 

##using mean treshold to make a minimum-level gray-scale mean image
arr = np.shape(gray_mean)  
for i in range(0,arr[0]):
    for j in range(0,arr[1]):
        if gray_mean[i,j] >= thresh:
            gray_mean[i,j] = 255
        else: 
            gray_mean[i,j] = 0
            
##print the threshold
print "mean threshod:", thresh                       
    
##save the results of mean threshold on the minimum-level gray-scale image
plt.savefig('/home/seirana/Documents/Results/threshold_mean.jpg') 

image=mpimg.imread('/home/seirana/Documents/coins.jpg')
imgplot = plt.imshow(image)
plt.show() 

image = img.convert("L")
arr = np.asarray(image)
plt.imshow(arr, cmap='gray')
plt.show()

img = cv2.imread('/home/seirana/Documents/two.jpg')       

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([image],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show() 

imgA = img.reshape(-1)
img = imgA   

ret,thresh = cv2.threshold(edited,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv2.moments(cnt)
print type(M)
print M
print "contours:", len(contours)
area = cv2.contourArea(cnt)
print area
epsilon = 0.01*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True) 
print type(approx)
print np.shape(approx)
#plt.show()
#cv2.imwrite('/home/seirana/Documents/Results/approx.jpg', cnt)

###normalizing data
#from sklearn import preprocessing
#normalized_X = preprocessing.normalize(X)

##cover.save('/home/seirana/Documents/Results/levelDimensionZerro.jpg', gray.format)

##load the counturs from a.npy file and print it
ra = np.load('/home/seirana/Documents/Results/counturs.npy')

