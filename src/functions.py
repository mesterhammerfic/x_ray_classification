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
    directory_iteratorgen = image.ImageDataGenerator(rescale=1./225)

    train_data = train_datagen.flow_from_directory('../../data/chest_xray/train/',
                                                   target_size=(100,100),
                                                   batch_size=32,
                                                   class_mode='binary',
                                                   color_mode='grayscale')
    val_data = val_datagen.flow_from_directory('../../data/chest_xray/val/',
                                                   target_size=(100,100),
                                                   batch_size=32,
                                                   class_mode='binary',
                                                   color_mode='grayscale')
    directory_iterator = directory_iteratorgen.flow_from_directory('../../data/chest_xray/test//',
                                                 target_size=(100,100),
                                                 batch_size=32,
                                                 class_mode='binary',
                                                 color_mode='grayscale')
    return train_data, val_data, directory_iterator



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
    sns.heatmap(cm,  cmap= 'Greens', annot=True, fmt='d')
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
    first_batch = directory_iterator.next()
    data_list = first_batch[0]
    batch_index = 0
    class_list = first_batch[1]
    while batch_index <= directory_iterator.batch_index:
        data = directory_iterator.next()
        data_list=np.concatenate((data_list, data[0]))
        class_list=np.concatenate((class_list, data[1]))
        batch_index = batch_index + 1


    # Second iteration creates full list
    first_batch = directory_iterator.next()
    data_list = first_batch[0]
    batch_index = 0
    class_list = first_batch[1]
    while batch_index <= directory_iterator.batch_index:
        data = directory_iterator.next()
        data_list=np.concatenate((data_list, data[0]))
        class_list=np.concatenate((class_list, data[1]))
        batch_index = batch_index + 1
    data_array = np.asarray(data_list)
    
    return data_array, class_list


def roc_auc_plotter(y, y_hat, title = 'ROC-AUC Curve'):
    """Function that takes in actual classifications
    and predicted classifications to return
    the roc-auc curve plot and score."""
    fpr, tpr, thresholds = roc_curve(y, y_hat)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linestyle='-', label='ROC-AUC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.show()
    print('ROC-AUC score:', roc_auc)
    
    
    
def load_final_data_generators():
    """
    Returns 3 keras image data generator objects that pull from the /data 
    folder of this directory. This function includes the ImageDataGenerator 
    parameters Horizontal flip, shear range, and zoom scale. We also 
    removed the grey scale color mode to use the images on pre-trained
    models.
    """
    
    train_datagen = image.ImageDataGenerator(horizontal_flip=True, 
                                         shear_range=0.2,
                                         zoom_range=0.3,
                                         rescale=1./225)
    val_datagen = image.ImageDataGenerator(rescale=1./225)
    directory_iteratorgen = image.ImageDataGenerator(rescale=1./225)

    train_data = train_datagen.flow_from_directory('../../data/chest_xray/train/',
                                               target_size=(100,100),
                                               batch_size=32,
                                               class_mode='binary')
    val_data = val_datagen.flow_from_directory('../../data/chest_xray/val/',
                                               target_size=(100,100),
                                               batch_size=32,
                                               class_mode='binary')
    directory_iterator = directory_iteratorgen.flow_from_directory('../../data/chest_xray/test//',
                                             target_size=(100,100),
                                             batch_size=32,
                                             class_mode='binary')
    return train_data, val_data, directory_iterator

def load_datagen_report():

    val_datagen = image.ImageDataGenerator(rescale=1./225)
    directory_iteratorgen = image.ImageDataGenerator(rescale=1./225)
    train_datagen = image.ImageDataGenerator(horizontal_flip=True, 
                                         shear_range=0.3,
                                         zoom_range=0.4,
                                         rescale=1./225)
    train_data = train_datagen.flow_from_directory('../../data/chest_xray/train/',
                                               target_size=(100,100),
                                               batch_size=32,
                                               class_mode='binary')
    val_data = val_datagen.flow_from_directory('../../data/chest_xray/val/',
                                               target_size=(100,100),
                                               batch_size=32,
                                               class_mode='binary')
    directory_iterator = directory_iteratorgen.flow_from_directory('../../data/chest_xray/test/',
                                             target_size=(100,100),
                                             batch_size=32,
                                             class_mode='binary')
    return train_data, val_data, directory_iterator


def plot_acc_loss(model, test_data, model_history):
    """This function takes in the model, the test
    data, and the model history to return the training
    and validation loss as well as a plot of
    both metrics."""
    train_loss = model_history.history['loss']
    train_acc = model_history.history['accuracy']
    val_loss = model_history.history['val_loss']
    val_acc = model_history.history['val_accuracy']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    sns.lineplot(model_history.epoch, train_loss, ax=ax1, label='train_loss')
    sns.lineplot(model_history.epoch, train_acc, ax=ax2, label='train_accuracy')
    sns.lineplot(model_history.epoch, val_loss, ax=ax1, label='val_loss')
    sns.lineplot(model_history.epoch, val_acc, ax=ax2, label='val_accuracy')
    eval = model.evaluate_generator(test_data, steps=len(test_data), verbose=1)
    return list(zip(model.metrics_names, eval))

def get_class_weight(train_files, val_files):
    """This function takes in training and validation
    files and returns the weight for each class
    of image."""
    count_normal = len(train_files[0]) + len(val_files[0])
    count_pneumonia = len(train_files[1]) + len(val_files[1])
    initial_bias = np.log([count_pneumonia/count_normal])
    weight_normal = (1 / count_normal) * (count_normal+count_pneumonia) / 2
    weight_pneumonia = (1 / count_pneumonia) * (count_normal+count_pneumonia) / 2
    class_weight = {0: weight_normal, 1: weight_pneumonia}
    print(class_weight)
    return class_weight
    
    


