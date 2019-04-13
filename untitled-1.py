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