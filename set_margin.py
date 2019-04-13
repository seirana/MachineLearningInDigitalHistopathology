'''
    This function draws margins around the convex hulls from all sildes and make a list of that margin
'''
import numpy as np

def convexhull_margins(addrs):
    slides = np.load(addrs+"slides") #slides is a nested dictionary
    all_white = 255*8
    margins = {} #save margin instead of convex hull
    for slide_th in slides.items():
        convex_hull = np.load(addrs+slides[slide_th]['slides_ID'] + "_convex_hull")
        mat = np.zeros(np.shape(convex_hull))
        #set a margin aroung the convex hull
        #slide is black=0 and covex hulls are white=255 then we will get a white margin
        for tissue_th in slides[slide_th]['tissues']:
            for w in range(slides[slide_th]['tissues'][tissue_th]['left'],slides[slide_th]['tissues'][tissue_th]['right']+1):
                for h in range(slides[slide_th]['tissues'][tissue_th]['up'],slides[slide_th]['tissues'][tissue_th]['down']+1):
                    mat[w][h] = 0
                    if convex_hull[w][h] == 255:
                        sm = 0
                        for k in range(-1,2):
                            for l in range(-1,2):
                                sm = sm + convex_hull[w+k][h+l]
                                
                        if sm == all_white: #then the point is inside the convex hull is a not a margin
                            mat[w][h] = 0
                        else:
                            mat[w][h] = 255 #then the point is a margin
                            
        convex_hull  = np.copy(mat)
        
        #save the margin as a list
        '''
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
        '''
        
        margin = list()
        for tissue_th in slides[slide_th]['tissues']:
            margin.append([slides[slide_th]['tissues'][tissue_th]['left'],slides[slide_th]['tissues'][tissue_th]['right']])
            for lr in range(slides[slide_th]['tissues'][tissue_th]['left'],slides[slide_th]['tissues'][tissue_th]['right']+1):
                for up in range(slides[slide_th]['tissues'][tissue_th]['up'],slides[slide_th]['tissues'][tissue_th]['down']+1):
                    up = slides[slide_th]['tissues'][tissue_th]['up']
                    while mat[lr][up] != 255:
                        up+=1
                    upperbound_h = up
                    
                    down = slides[slide_th]['tissues'][tissue_th]['down']
                    while mat[lr][down] != 255:
                        down-=1
                    lowerbound_h = down
                    
                    margin.append([upperbound_h,lowerbound_h])
            margins[tissue_th] = margin
        np.save(addrs+slides[slide_th]['tissues']+"_margin",  margins)
    return