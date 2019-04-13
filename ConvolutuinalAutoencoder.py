## Importing the modules
import os
os.system('clear')

from keras import Sequential
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import  RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
from train_test_random_list import make_rand_list
from fit_generator import patch_generator
from tissues_slides_list_percentage import tissues_slides_list
addrs = "/home/seirana/Desktop/Workstation/casp3/"
result_addrs = "/home/seirana/Desktop/Workstation/casp3/code/"
img_lst = np.load(addrs + "_SlidesInfo_dic.npy")[()]
patch_size = 256
inChannel = 3 #RGB images
x, y = patch_size, patch_size
input_img = Input(shape = (x, y, inChannel))

##Data Preprocessing
"""
Encoder: It has 5 Convolution blocks, each block has a convolution layer 
followed a batch normalization layer. 
Max-pooling layer is used after the first and second convolution blocks.
"""
##encoder
autoencoder = Sequential()

##The first convolution block
autoencoder.add(Conv2D(32,(3,3), activation='relu', padding='same', input_shape=(x, y, inChannel))) 
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(32,(3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(MaxPooling2D(pool_size=(2,2))) 

##The second convolution block
autoencoder.add(Conv2D(64,(3,3), activation='relu', padding='same')) 
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(64,(3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(MaxPooling2D(pool_size=(2, 2)))
    
##The third block
autoencoder.add(Conv2D(128,(3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(128,(3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(MaxPooling2D(pool_size=(2, 2)))

##The forth block
autoencoder.add(Conv2D(256,(3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(256,(3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(MaxPooling2D(pool_size=(2, 2)))
       
##The last block
autoencoder.add(Conv2D(32,(3,3), activation='relu', padding='same')) 
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(32,(3,3), activation='relu', padding='same', name='code'))

#decoder
"""
    Decoder: It has 2 Convolution blocks, each block has a convolution layer 
    followed a batch normalization layer. 
    Upsampling layer is used after the first and second convolution blocks.
"""
##The first block
autoencoder.add(Conv2D(64,(3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(64,(3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(UpSampling2D((2,2)))

##The second block
autoencoder.add(Conv2D(32,(3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(32,(3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(UpSampling2D((2,2)))

##The third block   
autoencoder.add(Conv2D(16,(3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(16,(3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(UpSampling2D((2,2)))
    
##The forthd block 
autoencoder.add(Conv2D(8,(3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(8,(3,3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(UpSampling2D((2,2)))

##The final layer of encoder will have 1 filter of size 3 x 3 which will reconstruct back the input having a single channel.
autoencoder.add(Conv2D(3,(3,3), activation='sigmoid', padding='same'))

'''begin'''
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

'''
call functions to generate train and validation data for each bach
'''
epochs_ = 1000 #number of epochs, we will have
batch_sz = 1000 #the batch siz,we we will take
train_per = 0.8 #the percentage of data, which will be used for training
validation_per = 1-train_per
all_patchs = 600000 #number of patches we want to extract of each image
steps_per_epoch_ = int(all_patchs * train_per / batch_sz) #TotalTrainingSamples / TrainingBatchSize
validation_steps_ = int(all_patchs * validation_per / batch_sz) #TotalvalidationSamples / ValidationBatchSize

'''
process the data
'''
##the summary function, this will show number of parameters (weights and biases) in each layer and also the total parameters in your model
autoencoder.summary()
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

resolution = 0
array_length = len(img_lst)
train_per = 0.8
#make_rand_list(addrs, len(img_lst), train_per)
#tissues_slides_list()
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=0, mode='min', baseline=None, restore_best_weights=False)
modelcheckpoint = ModelCheckpoint(result_addrs+'best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
autoencoder_train = autoencoder.fit_generator(patch_generator(addrs, patch_size, batch_sz, resolution, array_length, train_per, 'train'),\
                                              steps_per_epoch=steps_per_epoch_,\
                                              epochs=epochs_,\
                                              verbose=1,\
                                              callbacks=[earlystop, modelcheckpoint],\
                                              validation_data=patch_generator(addrs, patch_size, batch_sz, resolution, array_length, train_per, 'test'),\
                                              validation_steps=validation_steps_,\
                                              class_weight=None,\
                                              max_queue_size=10,\
                                              workers=1,\
                                              use_multiprocessing=True,\
                                              shuffle=True,\
                                              initial_epoch=0)
'''end'''

# load the saved model
saved_model = load_model(result_addrs+'best_model.h5')
# evaluate the model
##the summary function, this will show the number of parameters (weights and biases) in each layer and also the total parameters in the model
for layer in saved_model.layers:
    if layer.name == "code":
        print(layer.name)
        weights = layer.get_weights()
        np.save(result_addrs+"_code_weights", weights)
        output = layer.get_output()
        np.save(result_addrs+"_code_output", output)
        config=layer.get_config()
        np.save(result_addrs+"_code-config", config)

##plot the loss plot between training, summarize history for loss 
plt.figure()
plt.plot(autoencoder_train.history['loss'])
plt.plot(autoencoder_train.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(result_addrs + '_LossFunction.jpg')