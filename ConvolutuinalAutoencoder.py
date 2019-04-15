# importing the modules
import os
import numpy as np

# import tensorflow as tf
# import keras
from matplotlib import pyplot as plt
from keras import Sequential
# from keras import backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from train_test_random_list import make_rand_list
from fit_generator import patch_generator

os.system('clear')
# K.tensorflow_backend._get_available_gpus()
# config = tf.ConfigProto(device_count={'GPU': 2, 'CPU': 12})
# sess = tf.Session(config=config)
# K.set_session(sess)
# print("config:", config, "sess:", sess)

addrs = "/home/seirana/Workstation/casp3/"
result_addrs = "/home/seirana/Workstation/casp3/code/"
img_lst = np.load(addrs + "_SlidesInfo_dic.npy")[()]
patch_size = 256
inChannel = 3  # RGB images
x, y = patch_size, patch_size
input_img = Input(shape=(x, y, inChannel))

# Data Pre_processing
"""
Encoder: It has 5 Convolution blocks, each block has a convolution layer 
followed a batch normalization layer. 
Max-pooling layer is used after the first and second convolution blocks.
"""
# encoder
autoencoder = Sequential()

# The first convolution block
autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(x, y, inChannel)))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(MaxPooling2D(pool_size=(2, 2)))

# The second convolution block
autoencoder.add(Conv2D(48, (3, 3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(48, (3, 3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(MaxPooling2D(pool_size=(2, 2)))
    
# The third block
autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(MaxPooling2D(pool_size=(2, 2)))

# The forth block
autoencoder.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(MaxPooling2D(pool_size=(2, 2)))
       
# The last block
autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='code'))

# decoder
"""
    Decoder: It has 2 Convolution blocks, each block has a convolution layer 
    followed a batch normalization layer. 
    Upsampling layer is used after the first and second convolution blocks.
"""
# The first block
autoencoder.add(Conv2D(24, (3, 3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(24, (3, 3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(UpSampling2D((2, 2)))

# The second block
autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(UpSampling2D((2, 2)))

# The third block
autoencoder.add(Conv2D(12, (3, 3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(12, (3, 3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(UpSampling2D((2, 2)))
    
# The forth block
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
autoencoder.add(BatchNormalization())
autoencoder.add(UpSampling2D((2, 2)))

# The final layer
autoencoder.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

'''begin'''
'''
call functions to generate train and validation data for each bach
'''
epochs_ = 10000  # number of epochs, we will have
train_per = 0.8  # the percentage of data, which will be used for training
validation_per = round(1 - train_per, 1)

batch_sz = 100  # the batch siz,we we will take
all_patchs = 100000  # number of all patches we want to extract
steps_per_epoch_ = 1000  # int(batch_sz_train/batch_sz_train)  # TotalTrainingSamples / TrainingBatchSize
validation_steps_ = 1000  # int(batch_sz_test/batch_sz_test)  # TotalvalidationSamples / ValidationBatchSize
resolution = 0

'''
process the data
'''
'''
the summary function, this will show number of parameters (weights and biases) in each layer and
also the total parameters in your model
 
 '''
autoencoder.summary()
# keras.utils.multi_gpu_model(autoencoder, gpus=2, cpu_merge=True, cpu_relocation=False)
autoencoder.compile(loss='mean_squared_error', optimizer=RMSprop())

make_rand_list(addrs, len(img_lst), train_per)

early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=0.1,
                           patience=50,
                           verbose=0,
                           mode='min',
                           baseline=None,
                           restore_best_weights=True)

modelcheckpoint = ModelCheckpoint(result_addrs+'best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

autoencoder_train = autoencoder.fit_generator(patch_generator(addrs,
                                                              patch_size,
                                                              0.8*batch_sz,
                                                              resolution,
                                                              'train'),
                                              steps_per_epoch=steps_per_epoch_,
                                              epochs=epochs_,
                                              verbose=1,
                                              callbacks=[early_stop, modelcheckpoint],
                                              validation_data=patch_generator(addrs,
                                                                              patch_size,
                                                                              validation_per*batch_sz,
                                                                              resolution,
                                                                              'test'),
                                              validation_steps=validation_steps_,
                                              class_weight=None,
                                              max_queue_size=1,
                                              workers=1,
                                              use_multiprocessing=False,
                                              shuffle=True)  # initial_epoch=0)
'''end'''

'''load the saved model'''
saved_model = load_model(result_addrs+'best_model.h5')
# evaluate the model
'''
the summary function, this will show the number of parameters (weights and biases) in each 
layer and also the total parameters in the model
'''

for layer in saved_model.layers:
    if layer.name == "code":
        print(layer.name)
        weights = layer.get_weights()
        np.save(result_addrs+"_code_weights", weights)
        '''#output = layer.get_output()'''
        '''#np.save(result_addrs+"_code_output", output)'''
        config = layer.get_config()
        np.save(result_addrs+"_code-config", config)

'''plot the loss plot between training, summarize history for loss'''
plt.figure()
plt.plot(saved_model.history['loss'])
plt.plot(saved_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(result_addrs + '_LossFunction.jpg')
