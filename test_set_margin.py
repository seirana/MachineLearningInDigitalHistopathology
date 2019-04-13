import numpy as np

def convexhull_margins(convexhull_w, convexhull_h, convex_hull):
    #slide is white = 255 and covex hulls are black = 0 then we will get a black margin
    w = convexhull_w[1]- convexhull_w[0]+1
    h = convexhull_h[1]- convexhull_h[0]+1
    convex_hull = np.zeros(np.shape(1000,500))
    mat = np.zeros(np.shape(1000,500))
    
    for i in range(0,w):
        for j in range(0,h):
            mat[i][j] = 255
            if convex_hull[i][j] == 0:
                sm = 0
                for k in range(-1,2):
                    for l in range(-1,2):
                        sm = sm + convex_hull[i+k][j+l]
                        
                if sm == 0:
                    mat[i][j] = 255
                else:
                    mat[i][j] = 0
                    
    convex_hull  = np.copy(mat)                
    margin = list()
    margin.append([convexhull_w[0],convexhull_w[1]])
    for i in range(convexhull_w[0],convexhull_w[1]+1):
        j = convexhull_h[0]
        while mat[i][j] != 0:
            j+=1
        upperbound_h = j 
        
        j = convexhull_h[1]
        while mat[i][j] != 0:
            j-=1
        lowerbound_h = j
        margin.append([upperbound_h,lowerbound_h]) 
        np.save("/home/seirana/Documents/margin", margin)
    return