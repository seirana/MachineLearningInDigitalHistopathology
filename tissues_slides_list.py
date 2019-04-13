"""
    This function produces two lists of spaces that tissues cover the slides
    e.g. with 2 slides wi will get: tissues_slides_all_list = [[1,2],[1,2,3]] and tissues_slides_all_space_percentag = [[.1,.2],[.1,.2,.3]]
"""
import numpy as np

def tissues_slides_list(addrs):
    
    slides= np.load(addrs+"slides") #load a file to read information related to slides and tissue pieces on them

    resolution_level = slides[0][2] #the resolution of the slides that information comes from
    glassslide_w = np.floor((76*10^6)/((2^resolution_level)*226)) #the size of the  each glass slides is 76mmx26mm
    glassslide_h = np.floor((26*10^6)/((2^resolution_level)*226)) #the size of the  each glass slides is 76mmx26mm
    glassslide_space = glassslide_w * glassslide_h #the space size of the glass slide
    
    tissues_slides_all_list = list() #make a list of tissue spaces for all slides
    tissues_slides_all_space_percentage = list() #make a list of tissue spaces  percentage for all slides ##normilize the data  from tissues_slides_all_list 
    
    for i in range(0,len(slides)): #for all slides do
        convex_hull= np.load(addrs+slides[i][0] + "_convex_hull") #load the convex hull file for each slide
        tissues_per_slide_space_percentage = np.zeros(len(slides[i][6])) #percentage of coverage of tissues per slide
        tissues_per_slide_space_list = np.zeros(len(slides[i][6])) #make a list of tissue spaces for each slide
        
        for j in range(0,slides[i][6]):
            #calculate the space of tissue pieces for each slide
            for w in range(slides[i][7][j][0], slides[i][7][j][1]+1):
                for h in range(slides[i][7][j][2] - slides[i][7][j][3]+1):
                    if convex_hull[w][h] == 255: #if it is white=255 is inside the convex hull else it is black =0 then it is background
                        tissues_per_slide_space_list[j] += 1
            
            tissues_per_slide_space_percentage[j] = tissues_per_slide_space_list[j]/ glassslide_space #calculate the percentage of tissue coverage per slide
            
        tissues_slides_all_list.append(tissues_per_slide_space_list)
        tissues_slides_all_space_percentage.append(tissues_per_slide_space_percentage)
            
    np.save(addrs+"tissues_slides_space_percentage",tissues_slides_all_space_percentage)
    np.save(addrs+"tissues_slides_list",tissues_slides_all_list)
    return