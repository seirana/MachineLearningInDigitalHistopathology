"""
    This function produces two lists of spaces that tissues cover the slides
    e.g. with 2 slides wi will get: tissues_slides_all_list = [[1,2],[1,2,3]] and tissues_slides_all_space_percentag = [[.1,.2],[.1,.2,.3]]
"""
import numpy as np
import matplotlib.image as mpimg

def tissues_slides_list():
    addrs = "/home/seirana/Desktop/Workstation/casp3/"
    slides= np.load(addrs + "_SlidesInfo_dic.npy") #load a file to read information related to slides and tissue pieces on them
    resolution_level = slides.item().get(0)['magnification_level'] #the resolution of the slides that information comes from
    glassslide_w = np.floor((76*(10**5))/((2^resolution_level)*226)) #the size of the  each glass slides is 76mmx26mm
    glassslide_h = np.floor((26*10**5)/((2^resolution_level)*226)) #the size of the  each glass slides is 76mmx26mm
    glassslide_space = glassslide_w * glassslide_h #the space size of the glass slide
    
    tissues_slides_all_list = list() #make a list of tissue spaces for all slides
    tissues_slides_all_space_percentage = list() #make a list of tissue spaces  percentage for all slides ##normilize the data  from tissues_slides_all_list 
    
    for slide_th in slides.item(): #for all slides do
        convex_hull = mpimg.imread(addrs+slides.item().get(slide_th)['slide_ID'] + "_(5)convex_hull.jpg") #load the convex hull file for each slide
        tissues_per_slide_space_percentage = np.zeros(slides.item().get(slide_th)['tissue_count']) #percentage of coverage of tissues per slide
        tissues_per_slide_space_list = np.zeros(slides.item().get(slide_th)['tissue_count']) #make a list of tissue spaces for each slide
        
        for tissue_th in range(0,slides.item().get(slide_th)['tissue_count']):
            #calculate the space of tissue pieces for each slide
            for w in range(slides.item().get(slide_th)['tissues'][tissue_th]['left'],slides.item().get(slide_th)['tissues'][tissue_th]['right']):
                for h in range(slides.item().get(slide_th)['tissues'][tissue_th]['up'],slides.item().get(slide_th)['tissues'][tissue_th]['down']):
                    if convex_hull[h][w] == 255: #if it is white=255 is inside the convex hull else it is black =0 then it is background
                        tissues_per_slide_space_list[tissue_th] += 1
            
            tissues_per_slide_space_percentage[tissue_th] = tissues_per_slide_space_list[tissue_th]/ glassslide_space #calculate the percentage of tissue coverage per slide
            
        tissues_slides_all_list.append(tissues_per_slide_space_list)
        tissues_slides_all_space_percentage.append(tissues_per_slide_space_percentage)
       
    np.save(addrs+"tissues_slides_space_percentage",tissues_slides_all_space_percentage)
    np.save(addrs+"tissues_slides_list",tissues_slides_all_list)
    return