from tensorflow.keras.preprocessing import image
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_curve, auc, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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



def evaluation(y, y_hat, title = 'Confusion Matrix'):
    '''takes in true values and predicted values.
    The function then prints out a classifcation report
    as well as a confusion matrix using seaborn's heatmap.'''
    cm = confusion_matrix(y, y_hat)
    precision = precision_score(y, y_hat)
    recall = recall_score(y, y_hat)
    accuracy = accuracy_score(y,y_hat)
    print(classification_report(y, y_hat, target_names=['NORMAL', 'PNEUMONIA']))
    print('Accurancy: ', accuracy)
    sns.heatmap(cm,  cmap= 'Greens', annot=True)
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.title(title)
    plt.show()
    
def generator_to_array(directory_iterator):
    """
    input: the directory iterator returned by a generator's 'flow_from_directory' method.
    output: a numpy array of images and their associated classes
    """
    # first iteration resets test generator
    first_batch = test_data.next()
    data_list = first_batch[0]
    batch_index = 0
    class_list = first_batch[1]
    while batch_index <= test_data.batch_index:
        data = test_data.next()
        data_list=np.concatenate((data_list, data[0]))
        class_list=np.concatenate((class_list, data[1]))
        batch_index = batch_index + 1


    # Second iteration creates full list
    first_batch = test_data.next()
    data_list = first_batch[0]
    batch_index = 0
    class_list = first_batch[1]
    while batch_index <= test_data.batch_index:
        data = test_data.next()
        data_list=np.concatenate((data_list, data[0]))
        class_list=np.concatenate((class_list, data[1]))
        batch_index = batch_index + 1
    data_array = np.asarray(data_list)
    
    return data_array, class_list