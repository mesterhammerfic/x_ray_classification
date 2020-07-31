# Improving Radiology Efficiency with AI

An image classifcation project using x-ray data from mendelay.

## Goal
Create a tool that uses Convolutional Neural Networks to assist in diagnosing patients with pneumonia based on their x-rays. Our tool will be optimized for sensitivity to minimize the number of patients who go undiagnosed. This project is aimed at hospital administrators who are looking to increase the efficiency of their radiology departments and our final product will be built with ease-of-use in mind.

### Objectives
- Create a CNN that can detect pneumonia in an x-ray image
- Create an interface that would allow the user to input an image and recieve a diagnosis from the CNN

### Status
 
<h4 style="text-align:left;">Best Model</h4>
We achieved the best detection of Pneumonia in chest x-rays with the InceptionResNetV2 Convolutional Neural Network. This CNN has been trained on over a million images from the ImageNet database. If you would like to know more about the model, take a look over the documentation below. 

We chose to fine tune this model by training the layers in the model after the 750th layer. We also set the Early Stopping Callback parameter to a patience level of 14 and to restore the best weights.  

[InceptionResNetV2 Documentation](https://scisharp.github.io/Keras.NET/api/Keras.Applications.Inception.InceptionResNetV2.htm)


<h5 style="text-align:center;">Inception-ResNet-V2</h5>


|             | Recall | Precision | Accuracy | Support |                           |
|-------------|-----------|--------|----------|---------|---------------------------|
| **Normal** |   95%    |  84%  |   89%   |  232   |    
| **Pneumonia** |   84%    |  94%  |   89%   |  195   |   

<h5 style="text-align:center;">Test Validation</h5>

|                           |           |
|---------------------------|-----------|
|  **Recall**             |     94%    |   
|  **Precision**        |   84%    |  
|  **Accuracy**     |   89%    |  
| **Roc-Auc**       |   89%

User interface - 

### Table of Contents
<b>/data</b> - contains all data files used for modelling.

<b>/notebooks</b> - contains step by step descriptions of our process including data exploration and model iterations

<b>/src</b> - contains all scripts that are used in the notebooks/ and reports/ files

<b>/reports</b> - contains the final notebook that describes our findings

### Data
The data was originally downloaded from [this](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) kaggle dataset, which references a study published by Cell which can be found [here](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5). The dataset includes x-ray images from both healthy patients and patients with pneumonia. Both bacterial and viral pneumonia are represented and labeled in this data set. X-ray views are only either posterior or anterior chest views. The diognoses used to classify the images were reviewed by the original research team with the help of two expert physicians.

#### Note on Cleaning:

The images were found to have what look like medical equipment in the x-rays, such as electrodes, IV tubes, and catheters. The pneumonia positive x-rays had a far higher rate of these objects which could lead to a false sense of model performance. We deduced that the pneumonia-positive x-rays were drawn from patients who were either already being treated for pneumonia or were suffering from illnesses that may have led to pneumonia. LIME analysis also indicated that some of our neural networks learned to look for these medical devices when making their diagnoses. Due to time contraints, we elected to delete the images with medical devices from our training data, but not from our test and validation data. We left the images-with-devices in the test and validation data to ensure that our model would still be able to diagnose with those objects present.

#### Note on Validation Set:

The original validation set had only 16 images in it, which made it difficult to gauge model performance as it progressed through the epochs. Essentialy, the small validation size meant that for each incorrect classification the validation performance would rise or drop no less than 6%. We increased the validation set size by drawing from the training data, allowing us to more accuractely gauge the models performance through the epochs.

#### Download:

The data we are using can be found [here](https://www.dropbox.com/s/r23oastdde1v215/chest_xray.zip?dl=0) and should be unzipped into the [data](/data) folder in this projects main directory.

### Tools
 - Python 3.8
   - tensorflow
   - keras
   - matplotlib
   - seaborn
   - lime
 - Anaconda
 - JupyterLab
 
### Next Steps:
 - Make the model more transparent in order to troubleshoot errors in classification
 - See how well the model generalizes to other ailments identified by x-rays
 - Diagnose the type of pneumonia ie viral vs bacterial
 - Classify the severity of Pneumonia



#### Team Members:

Chum Mapa: chaminda.mapa@gmail.com

Syd Rothman: sydrothman@gmail.com

Jason Wong: jwong853@gmail.com

Maximilian Esterhammer-Fic: mesterhammerfic@gmail.com
