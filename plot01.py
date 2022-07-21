from skimage import img_as_float
from skimage.util import invert
from resizeimage import resizeimage


im = cv2.imread('/home/seirana/Documents/CMU-1.ndpi.jpg')
imgray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

print contours
np.save('/home/seirana/Documents/myfile', contours)

ra = np.load('/home/seirana/Documents/myfile.npy')
print ra

print len(contours)

cv2.drawContours(gray, contours, -1, (0,255,0), 3)
plt.show()


mat = [[255 for x in range(0,levelDimension[levelCount-1][0])] for y in range(0,levelDimension[levelCount-1][1])]

rot_edges = np.rot90(gray_scale)



sz = np.shape(gray_scale)  
if sz[0] > sz[1]: 
    gray_scale = np.rot90(gray_scale)
    
    

# we open the file for reading
fileObject = open(file_Name,'r')  
# load the object from the file into var b
b = pickle.load(fileObject)     



fileObject = open(result_addrs + "07A-D_MaLTT_Ctrl24h_Casp3_MaLTT_Ctrl42h_Casp3_07A-D - 2015-07-05 10.20.02.ndpi_patch_list",'r')  
# load the object from the file into var b
b = pickle.load(fileObject)
print b


#plt.figure()
#plt.hist(np.ndarray.flatten(gray_scale))

##remove small objects
rem_sm_obj = remove_small_objects(gray_scale,  0.01 * levelDimension[img_th][0] * 0.01 * levelDimension[img_th][1], connectivity=2)

for i in range(0, levelDimension[img_th][1]):
    for j in range(0, levelDimension[img_th][0]):
        if rem_sm_obj[i][j] == False:
            gray_scale[i][j] = 0
        else:
            gray_scale[i][j] = 255

gray_scale = np.copy(rem_sm_obj)


##save the image after removing the small objects 
cv2.imwrite(result_addrs + file + '_(7)rem_sm_obj.jpg', gray_scale)   


#apply all thesolds in the minimum-level gray-scale image                   
fig, ax = try_all_threshold(gray_scale, figsize=(10, 8), verbose=False)

#save  the results of all thesolds in the minimum-level gray-scale image as a .jpg file
fig.savefig(result_addrs + file + '_(2)try_all_threshold.jpg')


binary = gray_scale> thresh  

fig, axes = plt.subplots(ncols=2, figsize=(8, 3))
ax = axes.ravel()

ax[0].imshow(gray_scale, cmap=plt.cm.gray)
ax[0].set_title('Original image')

ax[1].imshow(binary, cmap=plt.cm.gray)
ax[1].set_title('Result')

for a in ax:
    a.axis('off') 
    
                
##save the results of isodata threshold on the minimum-level gray-scale image
plt.savefig(result_addrs + file + '_(3)isodata.jpg')     

##do canny edge detection on the minimum-level gray-scale isodata edited image
edges = cv2.Canny(gray_scale,1,254)
plt.subplot(121),plt.imshow(gray_scale,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
       

##save the image afted applying canny edge detection on the minimum-level gray-scale isodata edited image
gray_scale = np.copy(edges)
cv2.imwrite(result_addrs + file + '_(3)edges.jpg', gray_scale) 

for cnt in contours:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    print len(approx)
    if len(approx)==4:
        print "square"
        cv2.drawContours(img,[cnt],0,(0,0,255),-1)
        
        
        img_info.append([file_name, "d", levelCount, "d", levelDimension, "d", w1, "d", w2, "d", w3, "d", w4, "d", w5, "d", w6, "d", w7, "d"])

##save the information for the the images in a file
file_Name = result_addrs + "Images_INFO"
fileObject = open(file_Name,'wb') 
pickle.dump(img_info,fileObject)   
fileObject.close()
np.savetxt("/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/Results/AAA.csv", img_info, fmt='%s')         
        
