'''
define a new function
'''      
def image_info():
    ##define the new image size (resizing)
    inpt = input("Please insert the desired manginifasion layer:(between 0 and 5)")    
    sm = 0
    for ch in inpt:
        sm = sm + ord(ch)    
        
    ln1 = 0
    for ch in str(0):
        ln1 = ln1 + ord(ch)
    
    ln2 = 0
    for ch in str(5):
        ln2 = ln2 + ord(ch)
        
    while sm < ln1 or sm > ln2:       
        inpt = input("Please insert the desired manginifasion layer:(between 0 and 5)") 
        sm = 0
        for ch in inpt:
            sm = sm + ord(ch)   
   
    img_size = inpt
    
    ##define a patch size
    inpt = input("Please insert the desired patch size:(between 25 and 1000)")
    sm = 0
    for ch in inpt:
        sm = sm + ord(ch)
        
    ln1 = 0
    for ch in str(25):
        ln1 = ln1 + ord(ch)
    
    ln2 = 0
    for ch in str(1000):
        ln2 = ln2 + ord(ch)
        
    while sm < ln1 or sm > ln2:
        inpt = input("Please insert the desired patch size:(between 25 and 1000)")
        sm = 0
        for ch in inpt:
            sm = sm + ord(ch)
            
    patch_size = int(inpt)
    
    ##overlap size for the patches(horisental)
    inpt = input("Please insert the horisental_overlaping percentage:(between 0 and 99)")
    sm = 0
    for ch in inpt:
        sm = sm + ord(ch)
        
    ln1 = 0
    for ch in str(0):
        ln1 = ln1 + ord(ch)
    
    ln2 = 0
    for ch in str(99):
        ln2 = ln2 + ord(ch)
        
    while sm < ln1 or sm > ln2:
        inpt = input("Please insert the horisental_overlaping percentage:(between 0 and 99)")
        sm = 0
        for ch in inpt:
            sm = sm + ord(ch)
    
    hor_ovlap = int(int(inpt)*patch_size/100)
    
    if hor_ovlap == patch_size:
        hor_ovlap = hor_ovlap-1  
    
    ##overlap size for the patches(vertical)
    inpt = input("Please insert the vertical_overlaping percentage:(between 0 and 99) ")
    sm = 0
    for ch in inpt:
        sm = sm + ord(ch)
        
    ln1 = 0        
    for ch in str(0):
        ln1 = ln1 + ord(ch)
    
    ln2 = 0        
    for ch in str(99):
        ln2 = ln2 + ord(ch)
        
    while sm < ln1 or sm > ln2:        
        inpt = input("Please insert the vertical_overlaping percentage:(between 0 and 99) ") 
        sm = 0
        for ch in inpt:
            sm = sm + ord(ch)
    
    ver_ovlap = int(int(inpt)*patch_size/100)
       
    if ver_ovlap == patch_size:
        ver_ovlap = ver_ovlap-1

    ##minimum coverage of the convex hull by the patches, 255 is for white pixles  
    inpt = input("Please insert the vertical_overlaping percentage:(between 0 and 100) ")
    sm = 0
    for ch in inpt:
        sm = sm + ord(ch)
        
    ln1 = 0        
    for ch in str(1):
        ln1 = ln1 + ord(ch)
    
    ln2 = 0        
    for ch in str(100):
        ln2 = ln2 + ord(ch)
        
    while sm < ln1 or sm > ln2:        
        inpt = input("Please insert the vertical_overlaping percentage:(between 0 and 100) ") 
        sm = 0
        for ch in inpt:
            sm = sm + ord(ch)
      
    per = inpt / 100 * 255 ##minimum coverage of the convex hull by the patches, 255 is for white pixles    
   
    return img_size, patch_size, ver_ovlap, hor_ovlap, per