'''
    # Save
    dictionary = {'hello':'world'}
    np.save('my_file.npy', dictionary) 
    
    # Load
    read_dictionary = np.load('my_file.npy').item()
    print(read_dictionary['hello']) # displays "world"
    print(read_dictionary.item().get('hello'))
'''
import numpy as np

file_name_to_change = "_SlidesInfo_list"
addrs = "/home/seirana/Desktop/Workstation/casp3/"

files = np.load(addrs+file_name_to_change+'.npy')
dic = {}
for j in range(0,len(files)):    
    dic[j] = {'slide_ID':files[j][0],\
              'number_of_available_resolutions':files[j][1],\
              'available_resolution_list':files[j][2],\
              'magnification_level':files[j][3],\
              'tissue_count':files[j][4],\
              'tissues':{i:{'up':files[j][5][i][0],'down':files[j][5][i][1],'left':files[j][5][i][2],'right':files[j][5][i][3]} for i in range(0,files[j][4])}\
                }

np.save(addrs+"_SlidesInfo_dic.npy", dic)