"""
To read nifti format images, reconstructing them using convolutional autoencoder
To observe the effectiveness of the model, we will be testing the model on:
Unseen images,Noisy images and Use a qualitative metric:
Peak signal to noise ratio (PSNR) to evaluate the performance of the reconstructed images.

First, In the implementation of convolutional autoencoder: we will Fit the preprocessed data into the model,
visualize the training and validation loss plot, save the trained model and finally predict on the test
set.

Next, we'll will test the robustness of the pre-trained model by adding noise into the test images and
see how well the model performs quantitatively.

Finally, you will test the predictions using a quantitative metric peak signal-to-noise ratio (PSNR) and
measure the performance of the model.
"""

## Importing the modules
import os
os.system('clear')

from keras import Sequential
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import  RMSprop
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import math

##Loading the data
"""
load the images on the memory
"""
img_addrs = "/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/Test/"
file = "01A-D_MaLTT_Ther72h_Casp3_MaLTT_Ther72h_Casp3_01A-D - 2015-07-04 10.27.12_patch_list0.npy"

file_name = img_addrs + file        
patch_list = np.load(file_name)

"""
You will be using only RGB channel anit amit the . 
So, let's also see how to use only the center slices and load them.
"""

images = patch_list[:,:,:,0:3] #remove the alpha channel
shp = np.shape(images)
images = images.reshape(-1, shp[1],shp[2],shp[3])

##Data Preprocessing

"""
rescale the data with using max-min normalisation technique
"""
for i in range(0,3):
    m = np.max(images[:,:,:,i]) 
    mi = np.min(images[:,:,:,i])
    images[:,:,:,i] = (images[:,:,:,i] - mi) / (m - mi)

temp = np.zeros([shp[0],shp[1]-4,shp[2]-4,shp[3]]) #temp = np.zeros([shp[0],shp[1]+4,shp[2]+4,shp[3]]) 
temp[:,:,:,:] = images[:,2:shp[1]-2,2:shp[2]-2,:] #temp[:,2:shp[1]+2,2:shp[2]+2,:] = images
images = temp
shp = np.shape(images)

"""
In order for the model to generalize well, we split the data into two parts: a training and a 
validation set. We will train the model on 80% of the data and validate it on 20% of the remaining 
training data.
"""
##from sklearn.model_selection import train_test_split
train_X,valid_X,train_ground,valid_ground = train_test_split(images,images,test_size=0.2,random_state=13)

##Data Exploration
##Shapes of training set
print("Dataset (images) shape: {shape}".format(shape=images.shape))

##take a look at couple of the training and validation images in the dataset
plt.figure(figsize=[5,5])

##Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(train_X[0], (shp[1],shp[2],shp[3]))
curr_img = curr_img.tolist()
plt.imshow(curr_img, cmap = 'gray')

##Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(valid_X[0], (shp[1],shp[2],shp[3]))
curr_img = curr_img.tolist()
plt.imshow(curr_img, cmap = 'gray')

plt.savefig(img_addrs + 'tensorflow_images_(01_02)Train_Validation.jpg')

##The Convolutional Autoencoder!
batch_size_ = 256
epochs_ = 3
inChannel = 3
x, y = shp[1], shp[2]
input_img = Input(shape = (x, y, inChannel))

"""
Encoder: It has 3 Convolution blocks, each block has a convolution layer 
followed a batch normalization layer. 
Max-pooling layer is used after the first and second convolution blocks.
"""
##encoder
##The first convolution block will have 32 filters of size 3 x 3, followed by a downsampling (max-pooling) layer,
autoencoder = Sequential()
autoencoder.add(Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(x, y, inChannel))) 
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(32, (5,5), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(MaxPooling2D(pool_size=(2, 2)))
    
##The second block will have 64 filters of size 3 x 3, followed by another downsampling layer
autoencoder.add(Conv2D(64, (5,5), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(64, ((5,5)), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(MaxPooling2D(pool_size=(2, 2))) 
       
##The third block will have 128 filters of size 3 x 3, followed by another downsampling layer
autoencoder.add(Conv2D(128, (5,5), activation='relu', padding='same')) 
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(128, ((5,5)), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(MaxPooling2D(pool_size=(2, 2))) 
        
##The final block of encoder will have 32 filters of size 3 x 3
autoencoder.add(Conv2D(32, (5,5), activation='relu', padding='same')) 
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(32, (5,5), activation='relu', padding='same'))

#decoder
"""
    Decoder: It has 2 Convolution blocks, each block has a convolution layer 
    followed a batch normalization layer. 
    Upsampling layer is used after the first and second convolution blocks.
"""
##The first block will have 32 filters of size 3 x 3 followed by a upsampling layer
autoencoder.add(Conv2D(128, ((5,5)), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(128, ((5,5)), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(UpSampling2D((2,2)))

##The second block will have 16 filters of size 3 x 3 followed by a upsampling layer    
autoencoder.add(Conv2D(64, ((5,5)), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(64, ((5,5)), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(UpSampling2D((2,2)))
    
##The second block will have 8 filters of size 3 x 3 followed by a upsampling layer
autoencoder.add(Conv2D(32, (5,5), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(32, (5,5), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(UpSampling2D((2,2)))
    
##the summary function, this will show number of parameters (weights and biases) in each layer and also the total parameters in your model
autoencoder.summary()
    
##The final layer of encoder will have 1 filter of size 3 x 3 which will reconstruct back the input having a single channel.
autoencoder.add(Conv2D(3, (5,5), activation='sigmoid', padding='same', name="lastLayer"))
#last_layer = autoencoder.layers[-1]
#autoencoder = Model(inputs=input_img, outputs=last_layer(input_img))
autoencoder.compile(loss='mean_squared_logarithmic_error', optimizer = RMSprop())

##train the model with Keras' fit() function
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size_,epochs=epochs_,verbose=1,validation_data=(valid_X, valid_ground))

##plot the loss plot between training 
# summarize history for loss
plt.figure()
plt.plot(autoencoder_train.history['loss'])
plt.plot(autoencoder_train.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(img_addrs + 'tensorflow_images_(04)LossFunction.jpg')

##Predicting on Validation Data
##Since here you do not have a testing data. Let's use the validation data for predicting on the model that you trained just now
pred = autoencoder.predict(valid_X)
plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(5):
    plt.subplot(1, 5, i+1)
    curr_img = np.reshape(valid_ground[i], (shp[1],shp[2],shp[3]))
    curr_img = curr_img.tolist()
    plt.imshow(curr_img, cmap='gray')
    plt.savefig(img_addrs + 'tensorflow_images_(05)Prediction.jpg')
    
plt.figure(figsize=(20, 4))
print("Reconstruction of Test Images")
for i in range(5):
    plt.subplot(1, 5, i+1)
    curr_img = np.reshape(pred[i], (shp[1],shp[2],shp[3]))
    curr_img = curr_img.tolist()
    plt.imshow(curr_img, cmap='gray') 
    plt.savefig(img_addrs + 'tensorflow_images_(06)Reconstruction of Test Images.jpg')

##Predicting on Noisy images
##First let's add some noise into the validation images with a mean zero and standard deviation of 0.03.
[a,b,c,d]= np.shape(valid_X)
mean = 0
sigma = 0.03
gauss = np.random.normal(mean,sigma,(a,b,c,d))
noisy_images = valid_X + gauss

pred_noisy = autoencoder.predict(noisy_images)
plt.figure(figsize=(20, 4))
print("Noisy Test Images")
for i in range(5):
    plt.subplot(1, 5, i+1)
    curr_img = np.reshape(noisy_images[i], (shp[1],shp[2],shp[3]))
    curr_img = curr_img.tolist()
    plt.imshow(curr_img, cmap='gray')
    plt.savefig(img_addrs + 'tensorflow_images_(07)Noisy Images.jpg')
  
plt.figure(figsize=(20, 4))
print("Reconstruction of Noisy Test Images")
for i in range(5):
    plt.subplot(1, 5, i+1)
    curr_img = np.reshape(pred_noisy[i], (shp[1],shp[2],shp[3]))
    curr_img = curr_img.tolist()
    plt.imshow(curr_img, cmap='gray') 
    plt.savefig(img_addrs + 'tensorflow_images_(08)Reconstruction of Noisy Images.jpg')

##Quantitative Metric: Peak Signal-to-Noise Ratio (PSNR)
"""
The PSNR block computes the peak signal-to-noise ratio, in decibels(dB), between two images. 
This ratio is often used as a quality measurement between the original and a reconstructed image. 
The higher the PSNR, the better the quality of the reconstructed image.
"""
valid_pred = autoencoder.predict(valid_X)
mse =  np.mean((valid_X - valid_pred) ** 2)
psnr = 20 * math.log10( 1.0 / math.sqrt(mse))
print('PSNR of reconstructed validation images: {psnr}dB'.format(psnr=np.round(psnr,2)))

noisy_pred = autoencoder.predict(noisy_images)
mse =  np.mean((valid_X - noisy_pred) ** 2)
psnr_noisy = 20 * math.log10( 1.0 / math.sqrt(mse))
print('PSNR of reconstructed noisy images: {psnr}dB'.format(psnr=np.round(psnr_noisy,2)))
#################################################
"""
The history object is returned from calls to the fit() function used to train the model. 
Metrics are stored in a dictionary in the history member of the object returned.
"""
img_addrs = "/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/Test/"
test_file = "01A-D_MaLTT_Ther72h_F4-80_MaLTT_Ther72h_F4-80_01A-D - 2015-07-04 10.38.21_patch_list0.npy"
test_file_name = img_addrs + test_file        
test_patch_list = np.load(test_file_name)

test_images = test_patch_list[:,:,:,0:3] 
test_shp = np.shape(test_images)
test_images = test_images.reshape(-1, test_shp[1],test_shp[2],3)

for i in range(0,3):
    m = np.max(test_images[:,:,:,i]) 
    mi = np.min(test_images[:,:,:,i])
    test_images[:,:,:,i] = (test_images[:,:,:,i] - mi) / (m - mi)

temp = np.zeros([test_shp[0],test_shp[1]-4,test_shp[2]-4,test_shp[3]]) 
temp[:,:,:,:] = test_images[:,2:test_shp[1]-2,2:test_shp[1]-2,:] 
test_images = temp

pred_noisy = autoencoder.predict(test_images)
test_pred = autoencoder.predict(test_images)
mse =  np.mean((test_images - test_pred) ** 2)
psnr_test = 20 * math.log10( 1.0 / math.sqrt(mse))

print('PSNR of reconstructed test images: {psnr}dB'.format(psnr=np.round(psnr_test,2)))

#################################################
"""
The history object is returned from calls to the fit() function used to train the model. 
Metrics are stored in a dictionary in the history member of the object returned.
"""
img_addrs = "/home/seirana/Disselhorst_Jonathan/MaLTT/Immunohistochemistry/Test/"
test_file = "31A-D_MaLTT_Ctrl24h_Casp3_2016-06-07_patch_list0.npy"
test_file_name = img_addrs + test_file        
test_patch_list = np.load(test_file_name)

test_images = test_patch_list[:,:,:,0:3] 
test_shp = np.shape(test_images)
test_images = test_images.reshape(-1, test_shp[1],test_shp[2],3)

for i in range(0,3):
    m = np.max(test_images[:,:,:,i]) 
    mi = np.min(test_images[:,:,:,i])
    test_images[:,:,:,i] = (test_images[:,:,:,i] - mi) / (m - mi)

#temp = np.zeros([test_shp[0],test_shp[1]+4,test_shp[2]+4,test_shp[3]])
#temp[:,2:test_shp[1]+2,2:test_shp[2]+2,:] = test_images
temp = np.zeros([test_shp[0],test_shp[1]-4,test_shp[2]-4,test_shp[3]]) 
temp[:,:,:,:] = test_images[:,2:test_shp[1]-2,2:test_shp[1]-2,:] 
test_images = temp

pred_noisy = autoencoder.predict(test_images)
test_pred = autoencoder.predict(test_images)
mse =  np.mean((test_images - test_pred) ** 2)
psnr_test = 20 * math.log10( 1.0 / math.sqrt(mse))
print('PSNR of reconstructed test images: {psnr}dB'.format(psnr=np.round(psnr_test,2)))

plt.show()
