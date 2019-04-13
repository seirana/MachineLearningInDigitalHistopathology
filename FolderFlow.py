from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(\
        rescale=1./255,\
        shear_range=0.2,\
        zoom_range=0.2,\
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
        
train_generator = train_datagen.flow_from_directory(\
    directory= img_addrs + "train/",\
    target_size=(shp[1], shp[2]),\
    color_mode="rgb",\
    batch_size=batch_size,\
    class_mode="input",\
    shuffle=True,\
    seed=42\
)

valid_generator = valid_datagen.flow_from_directory(\
    directory= img_addrs + "valid/",\
    target_size=(shp[1], shp[2]),\
    color_mode="rgb",\
    batch_size=batch_size,\
    class_mode="input",\
    shuffle=True,\
    seed=42\
)

test_generator = test_datagen.flow_from_directory(\
    directory= img_addrs + "test/",\
    target_size=(shp[1], shp[2]),\
    color_mode="rgb",\
    batch_size=1,\
    class_mode=None,\
    shuffle=False,\
    seed=42\
)        