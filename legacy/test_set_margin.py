'''
    This function draws margins around the convex hulls from all sildes and make a list of that margin
'''
import numpy as np

def convexhull_margins(addrs):
    slides = np.load(addrs+"slides")
    all_white = 255*8 
    for i in range(0,len(slides)):        
        convex_hull = np.load(addrs+slides[i][0] + "_convex_hull")
        mat = np.zeros(np.shape(convex_hull))
        #set a margin aroung the convex hull
        #slide is black=0 and covex hulls are white=255 then we will get a white margin
        for j in range(0,slides[i][6]):
            for w in range(slides[i][7][j][0], slides[i][7][j][1]+1):
                for h in range(slides[i][7][j][2] - slides[i][7][j][3]+1):
                    mat[i][j] = 0
                    if convex_hull[i][j] == 255:
                        sm = 0
                        for k in range(-1,2):
                            for l in range(-1,2):
                                sm = sm + convex_hull[i+k][j+l]
                                
                        if sm == all_white: #then the point is inside the convex hull is a not a margin
                            mat[i][j] = 0 
                        else:
                            mat[i][j] = 255 #then the point is a margin
                            
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
        margin.append([slides[i][7][j][0],slides[i][7][j][1]])
        for j in range(slides[i][7][j][0], slides[i][7][j][1]+1):
            k = slides[i][7][j][2]
            while mat[j][k] != 255:
                k+=1
            upperbound_h = k 
            
            k = slides[i][7][j][3]
            while mat[j][k] != 255:
                k-=1
            lowerbound_h = k 
            
            margin.append([upperbound_h,lowerbound_h]) 
            
        np.save(addrs+slides[i][0]+"_margin",  margin)
    return