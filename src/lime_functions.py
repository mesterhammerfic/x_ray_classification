import os
import sys
module_path = os.path.abspath(os.path.join(os.pardir, os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from tensorflow.python.keras.preprocessing import image

from skimage.io import imread
from skimage.segmentation import mark_boundaries

import lime
from lime import lime_image

def get_paths(directory):
    """
    Parameter
    ----------
    directory : path to root folder containing pneumonia xrays and normal xrays
    
    Returns
    --------
    - list of paths to all xray images in the directory
    - list of the xray's corresponding label (1 = 'PNEUMONIA', 0 = 'NORMAL')
    - DataFrame with column of image paths and column of labels
    
    Example
    --------
    train_dir = '../data/chest_xray/train'
    train_path_list, train_labels, train_df = get_paths(train_dir)
    """
    pneu_list = os.listdir(directory + '/PNEUMONIA')
    norm_list = os.listdir(directory + '/NORMAL')

    pneu_labels = [1 for img in pneu_list]
    norm_labels = [0 for img in norm_list]
    
    labels = pneu_labels + norm_labels

    pneu_path = [train_dir + '/PNEUMONIA/'+ img_id for img_id in pneu_list]
    norm_path = [train_dir + '/NORMAL/'+ img_id for img_id in norm_list]
    path_list = pneu_path + norm_path
    
    directory_df = pd.DataFrame(path_list, columns=['image_path'])
    directory_df['image_label']=labels
    return path_list, labels, directory_df

def preprocess_image(path_list):
    output = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = np.divide(x, 255.0)
        output.append(x)
    return np.vstack(output)
