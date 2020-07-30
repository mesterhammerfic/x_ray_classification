# x_ray_classification
An image classifcation project using x-ray data from mendelay.


### Data
The data was originally downloaded from [this](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) kaggle dataset, which references a study published by Cell which can be found [here](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5). The dataset includes x-ray images from both healthy patients and patients with pneumonia. Both bacterial and viral pneumonia are represented and labeled in this data set. X-ray views are only either posterior or anterior chest views. The diognoses used to classify the images were reviewed by the original research team with the help of two expert physicians.

#### Note on Cleaning:

The images were found to have what look like medical equipment in the x-rays, such as electrodes, IV tubes, and catheters. The pneumonia positive x-rays had a far higher rate of these objects which could lead to a false sense of model performance. We deduced that the pneumonia-positive x-rays were drawn from patients who were either already being treated for pneumonia or were suffering from illnesses that may have led to pneumonia. LIME analysis also indicated that some of our neural networks learned to look for these medical devices when making their diagnoses. Due to time contraints, we elected to delete the images with medical devices from our training data, but not from our test and validation data. We left the images-with-devices in the test and validation data to ensure that our model would still be able to diagnose with those objects present.

#### Note on Validation Set:

The original validation set had only 16 images in it, which made it difficult to gauge model performance as it progressed through the epochs. Essentialy, the small validation size meant that for each incorrect classification the validation performance would rise or drop no less than 6%. We increased the validation set size by drawing from the training data, allowing us to more accuractely gauge the models performance through the epochs.

Download:
The data we are using can be found [here](https://www.dropbox.com/s/r23oastdde1v215/chest_xray.zip?dl=0) and should be unzipped into the [data](/data) folder in this projects main directory.
