"""
    This function draws margins around the convex hulls from all sildes and make a list of that margin
"""
import numpy as np
import matplotlib.image as mpimg
import cv2

# def convexhull_margins(addrs):
addrs = "/home/seirana/Desktop/Workstation/casp3/"
slides = np.load(addrs+"_SlidesInfo_dic.npy")  # slides is a nested dictionary
all_white = 255*9
for slide_th in slides.item():  
    convex_hull = mpimg.imread(addrs+slide_th['slide_ID'] + "_(5)convex_hull.jpg")
    mat = np.zeros(np.shape(convex_hull))
    
    # set a margin around the convex hull
    # slide is black=0 and covex hulls are white=255 then we will get a white margin
    for tissue_th in slide_th['tissues']:
        for h in range(slide_th['tissues'][tissue_th]['up'], 
                       slide_th['tissues'][tissue_th]['down']+1):
            for w in range(slide_th['tissues'][tissue_th]['left'], 
                           slide_th['tissues'][tissue_th]['right']+1):
                if convex_hull[h][w] == 255:
                   if slide_th['tissues'][tissue_th]['up'] < h < \
                           slide_th['tissues'][tissue_th]['down']:
                       if slide_th['tissues'][tissue_th]['left'] < w < \
                               slide_th['tissues'][tissue_th]['right']:
                            sm = 0
                            for k in range(-1, 2):
                                for l in range(-1, 2):
                                    sm += convex_hull[h+k][w+l]

                            if sm != all_white:  # then the point is inside the convex hull, is a not a margin
                                mat[h][w] = 255  # then the point is a margin
                   if h == slide_th['tissues'][tissue_th]['up'] or \
                           h == slide_th['tissues'][tissue_th]['down']:
                       mat[h][w] = 255  # then the point is a margin
                   if w == slide_th['tissues'][tissue_th]['left'] or \
                           w == slide_th['tissues'][tissue_th]['right']:
                       mat[h][w] = 255  # then the point is a margin
     
    convex_hull = np.copy(mat)
    file = slide_th['slide_ID']
    cv2.imwrite(addrs + file + '_(6)margins.jpg', convex_hull)
    # save the margin as a list
    """
    if the margin is sth like that:
        
        (6,2)*********(6,11)
             *       ***
             *         *
           * *         ****
           *              *
      (1,1)****************(16,1)
      
    then this shape is between lines x = 1(x_left) and x = 16(x_right)
    for x=1 y is between 1 and 3, for x = 2, 1 <= y >= 6 and also the same from 3 to 11 we have 1 <= y >= 6
    for x = 12,13, 1 <= y >= 5, for x = 14,15 and 16 1 <= y >= 3
    
    we will the margine set like that {
            (1,16)"lines to surrond the convex hull", (1,3)"upper and lower bound for x =1",
            (1,6),(1,6),(1,6)(1,6)(1,6)(1,6)(1,6)(1,6)(1,6)(1,6)"upper and lower bound for x =2 to x = 11"
            (1,5),(1,5),(1,3),(1,3)}
    the length of the list = x_right - x_left + 2
    """
    margins = {}  # save margin instead of convex hull
    for tissue_th in slide_th['tissues']:
        margin = list()
        margin.append([slide_th['tissues'][tissue_th]['left'],
                       slide_th['tissues'][tissue_th]['right']])
        for lr in range(slide_th['tissues'][tissue_th]['left'],
                        slide_th['tissues'][tissue_th]['right']+1):
            
            up = slide_th['tissues'][tissue_th]['up']
            down = slide_th['tissues'][tissue_th]['down']
            
            while mat[up][lr] != 255 and up < down-1:
                up += 1
            if up == slide_th['tissues'][tissue_th]['down']: 
                upperbound = slide_th['tissues'][tissue_th]['up'] 
            else:
                upperbound = up
            
            down = slide_th['tissues'][tissue_th]['down']
            while mat[down][lr] != 255 and down > up + 1:
                down -= 1
            if down == slide_th['tissues'][tissue_th]['up']:
                upperbound = slide_th['tissues'][tissue_th]['down']
            else:
                lowerbound = down
                
            margin.append([upperbound, lowerbound])
    
        margins[tissue_th] = margin
        
    file = slide_th['slide_ID']
    np.save(addrs+file+"_margin.npy",  margins)
