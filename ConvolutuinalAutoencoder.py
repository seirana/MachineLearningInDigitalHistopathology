"""
To read nifti format images, reconstructing them using convolutional autoencoder
To observe the effectiveness of the model, we will be testing the model on unseen and noisy images.

First, In the implementation of convolutional autoencoder: we will Fit the preprocessed data into the model,
visualize the training and validation loss plot, save the the weights of code layer from trained model and finally predict on the test
set.

Next, we'll will test the robustness of the pre-trained model by adding noise into the test images and
see how well the model performs quantitatively.
"""

## Importing the modules
import os
os.system('clear')

from keras import Sequential#, Model
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import  RMSprop
from preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import math
from patch_production import patch_prod
from patch_batch_generator import patches_for_batches

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

temp = np.zeros([shp[0],shp[1]-4,shp[2]-4,shp[3]])  
temp[:,:,:,:] = images[:,2:shp[1]-2,2:shp[2]-2,:]
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
epochs_ = 2
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
autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(x, y, inChannel)))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(32, (3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(MaxPooling2D(pool_size=(2, 2)))
    
##The second block will have 64 filters of size 3 x 3, followed by another downsampling layer
autoencoder.add(Conv2D(64, (3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(64, ((3,3)), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(MaxPooling2D(pool_size=(2, 2)))
       
##The third block will have 128 filters of size 3 x 3, followed by another downsampling layer
autoencoder.add(Conv2D(128, (3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(128, ((3,3)), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(MaxPooling2D(pool_size=(2, 2)))
        
##The forth block of encoder will have 32 filters of size 3 x 3
autoencoder.add(Conv2D(256, (3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(256, (3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(MaxPooling2D(pool_size=(2, 2)))

##The final block of encoder will have 32 filters of size 3 x 3
autoencoder.add(Conv2D(32, (3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(32, (3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization(name = "code"))

#decoder
"""
    Decoder: It has 2 Convolution blocks, each block has a convolution layer 
    followed a batch normalization layer. 
    Upsampling layer is used after the first and second convolution blocks.
"""
##The first block will have 32 filters of size 3 x 3 followed by a upsampling layer
autoencoder.add(Conv2D(256, ((3,3)), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(256, ((3,3)), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(UpSampling2D((2,2)))

##The second block will have 16 filters of size 3 x 3 followed by a upsampling layer    
autoencoder.add(Conv2D(128, ((3,3)), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(128, ((3,3)), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(UpSampling2D((2,2)))
    
##The second block will have 8 filters of size 3 x 3 followed by a upsampling layer
autoencoder.add(Conv2D(64, (3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(64, (3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(UpSampling2D((2,2)))
    
##The third block will have 4 filters of size 3 x 3 followed by another upsampling layer
autoencoder.add(Conv2D(32, (3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(32, (3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(UpSampling2D((2,2)))
   
##The final layer of encoder will have 1 filter of size 3 x 3 which will reconstruct back the input having a single channel.
autoencoder.add(Conv2D(3, (3,3), activation='sigmoid', padding='same', name="lastLayer"))

##the summary function, this will show number of parameters (weights and biases) in each layer and also the total parameters in your model
autoencoder.summary()

autoencoder.compile(loss='mean_squared_logarithmic_error', optimizer = RMSprop()) #autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

##train the model with Keras' fit() function
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size_,epochs=epochs_,verbose=1,validation_data=(valid_X, valid_ground))

'''inprogress_begin'''
result_addrs = ""
information = ""
patch_num = ""

input_data = patches_for_batches(patch_num, information, result_addrs)

batch_sz = ""
x_train = ""
epochs_ = ""

'''
The generator is run in parallel to the model, for efficiency. For instance, 
    this allows you to do real-time data augmentation on images on CPU in parallel 
    to training your model on GPU.
    The use of keras.utils.Sequence guarantees the ordering and guarantees the single
    use of every input per epoch when using use_multiprocessing=True.

generator: A generator or an instance of Sequence (keras.utils.Sequence) object 
    in order to avoid duplicate data when using multiprocessing. The output of the 
    generator must be either
    a tuple (inputs, targets)
    a tuple (inputs, targets, sample_weights).
    This tuple (a single output of the generator) makes a single batch. 
    Therefore, all arrays in this tuple must have the same length (equal to the size of this batch). 
    Different batches may have different sizes. For example, the last batch of the epoch is commonly
    smaller than the others, if the size of the dataset is not divisible by the batch size. 
    the generator is expected to loop over its data indefinitely. An epoch finishes 
    when steps_per_epoch batches have been seen by the model.

steps_per_epoch: Integer. Total number of steps (batches of samples) to yield from generator
    before declaring one epoch finished and starting the next epoch. It should typically be equal
    to the number of samples of your dataset divided by the batch size. Optional for Sequence:
    if unspecified, will use the len(generator) as a number of steps.
    
validation_data: This can be either
    a generator or a Sequence object for the validation data
    tuple (x_val, y_val)
    tuple (x_val, y_val, val_sample_weights)
    on which to evaluate the loss and any model metrics at the end of each epoch. 
    The model will not be trained on this data.    
    
validation_steps: Only relevant if validation_data is a generator. 
    Total number of steps (batches of samples) to yield from validation_data generator 
    before stopping at the end of every epoch. It should typically be equal to the 
    number of samples of your validation dataset divided by the batch size. 
    optional for Sequence: if unspecified, will use the len(validation_data) as a number of steps. 
    
max_queue_size: Integer. Maximum size for the generator queue. 
    If unspecified, max_queue_size will default to 10.
    
workers: Integer. Maximum number of processes to spin up when using process-based threading. 
    If unspecified, workers will default to 1. If 0, will execute the generator on the main thread.
    
use_multiprocessing: Boolean. If True, use process-based threading. 
    If unspecified, use_multiprocessing will default to False. Note that because 
    this implementation relies on multiprocessing, you should not pass non-picklable 
    arguments to the generator as they can't be passed easily to children processes.    
'''
patch_prod(img_addrs, result_addrs, file, img_size, patch_size, ver_ovlap, hor_ovlap, per, information)

def generate_arrays_from_file(path):
    while 1:
    f = open(path)
    for line in f:
        # create numpy arrays of input data
        # and labels, from each line in the file
        x, y = process_line(line)
        yield x, y
    f.close() 
    
autoencoder_train = autoencoder.fit_generator(input_data, steps_per_epoch=len(x_train) / batch_sz, \
                                        epochs=epochs_, verbose=1, callbacks=None, validation_data=None, \
                                        validation_steps=None, class_weight=None, max_queue_size=10, workers=1, \
                                        use_multiprocessing=False, shuffle=True, initial_epoch=0)

predict_on_batch(X) #Returns predictions for a single batch of samples.
predict_proba(X, batch_size=128, verbose=1) #Generate class probability predictions for the input samples batch by batch.
test_on_batch(X, y, accuracy=False, sample_weight=None) #Returns the loss over a single batch of samples, or a tuple (loss, accuracy) if accuracy=True.
##train_on_batch(X, y, accuracy=False, class_weight=None, sample_weight=None) #Single gradient update over one batch of samples. Returns the loss over the data, or a tuple (loss, accuracy) if accuracy=True.
evaluate_generator(generator, nb_val_samples, show_accuracy=False, verbose=1)
predict_generator(generator, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
'''
Evaluates the model on a generator. The generator should return the same kind of 
    data with every yield as accepted by evaluate.
    If show_accuracy, it returns a tuple (loss, accuracy), otherwise it returns the loss value.
    Arguments:
            generator: generator yielding dictionaries of the kind accepted by evaluate, 
            or tuples of such dictionaries and associated dictionaries of sample weights.
            nb_val_samples: total number of samples to generate from generator to use in validation.
            show_accuracy: whether to log accuracy. Can only be used if your Graph has a single output
            (otherwise "accuracy" is ill-defined).
'''
'''inprogress_end'''

#the summary function, this will show number of parameters (weights and biases) in each layer and also the total parameters in your model
#for layer in autoencoder.layers:
#    if layer.name == "code" or layer.name == "conv2d_10":
#        print(layer)
#        print(layer.name)
#        weights = layer.get_weights()
#        output_ = layer.get_output()
#        print (len(weights[0]))
#        g=layer.get_config()
#        print(g)

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

temp = np.zeros([test_shp[0],test_shp[1]-4,test_shp[2]-4,test_shp[3]]) 
temp[:,:,:,:] = test_images[:,2:test_shp[1]-2,2:test_shp[1]-2,:] 
test_images = temp

test_pred = autoencoder.predict(test_images)
mse =  np.mean((test_images - test_pred) ** 2)
psnr_test = 20 * math.log10( 1.0 / math.sqrt(mse))
print('PSNR of reconstructed test images: {psnr}dB'.format(psnr=np.round(psnr_test,2)))

plt.show()