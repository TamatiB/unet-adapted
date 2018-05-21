# U-Net

ADAPTED from https://github.com/zhixuhao/unet

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

---

## Overview

### Data

[Provided data](http://brainiac2.mit.edu/isbi_challenge/) you can download the train and test data from this server.
Data flder hierachy is as follows

Data
    test
    train
        image
        label
        
### Pre-processing

Images stacks need to be in separate image tiffs with corresponding label names and organised according to the above hierachy.
The data for training contains 30 512*512 images, so image augmentation is perfomed implemented using the following [image deformation](http://faculty.cs.tamu.edu/schaefer/research/mls.pdf) method, in C++ using opencv which results in augmented images and accompanying label images. Which are saved as tiff image files in 

Data  -> aug_label
      -> aug_train
      
Images are then combined and saved in npy format in

Data  -> npydata
          -> imgs_mask_train.npy
          -> imgs_test.py
          -> imgs_train.py

### Model

This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

Output from the network is a 512*512 which represents mask that should be learned. Sigmoid activation function
makes sure that mask pixels are in \[0, 1\] range.

### Training

The model is trained for 10 epochs, on augmented data, with an automatic batch size of 4
Loss function for the training is basically just a binary crossentropy

---

## How to use

### Dependencies

This tutorial depends on the following libraries:

* Tensorflow
* Keras >= 1.0
* libtiff(optional)

Code is compatible with Python versions 3.5.

### Prepare the data

First transfer 3D volume tiff to 30 512*512 images.
Perfom image augmentation using Augmentation class
Create npy files for use by unet

this is done by running ```data.py```

### Define the model

* Check out ```get_unet()``` in ```unet.py``` to modify the model, optimizer and loss function.
* Unet-preset values are found in 

### Train the model and generate masks for test images

* Run ```python unet.py``` to train the model.

Model will be saved in ```unet.hdf5``` file for late use.

After this script finishes, in ```imgs_mask_test.npy``` masks for corresponding images in ```imgs_test.npy```
should be generated. I suggest you examine these masks for getting further insight of your model's performance.

### Results

Use the trained model to do segmentation on test images, the result is statisfactory.

