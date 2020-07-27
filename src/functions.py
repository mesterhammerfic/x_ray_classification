from tensorflow.keras.preprocessing import image

def load_data_generators():
    """
    Returns 3 keras image data generator objects that pull from the /data 
    folder of this directory.
    """
    train_datagen = image.ImageDataGenerator(horizontal_flip=True, 
                                             zoom_range=0.2,
                                             rescale=1./225)
    val_datagen = image.ImageDataGenerator(rescale=1./225)
    test_datagen = image.ImageDataGenerator(rescale=1./225)

    train_data = train_datagen.flow_from_directory('../data/chest_xray/train/',
                                                   target_size=(100,100),
                                                   batch_size=32,
                                                   class_mode='binary',
                                                   color_mode='grayscale')
    val_data = val_datagen.flow_from_directory('../data/chest_xray/val/',
                                                   target_size=(100,100),
                                                   batch_size=32,
                                                   class_mode='binary',
                                                   color_mode='grayscale')
    test_data = test_datagen.flow_from_directory('../data/chest_xray/test//',
                                                 target_size=(100,100),
                                                 batch_size=32,
                                                 class_mode='binary',
                                                 color_mode='grayscale')
    return train_data, val_data, test_data