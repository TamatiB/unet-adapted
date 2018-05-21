from keras.preprocessing.image import ImageDataGenerator, array_to_img
from scipy.misc import toimage, imsave
import numpy as np
from unet import *
# so this file was made to see if our npy files are in fact saving and loading
# images correctly because, these images are blank af.



# This is the train set
#datapath = "data/npydata/imgs_test.npy"
# This is the train set masks
#datapath = "data/npydata/imgs_mask_train.npy"

#Both of these work and load images nicely, so does the test set
datapath = "imgs_mask_test.npy"
# but the test masks that appear after training and testing are empty

def draw_npy_img(path):

    img_arrays = np.load(path)
    pil_img_array = []
    for img in img_arrays:
        pil_img = array_to_img(img)
        pil_img_array.append(pil_img)
        # either we save here or we see here

    pil_img_array[0].show()

datapath = "data/npydata/imgs_train.npy"
img_arrays = np.load(datapath)
print(len(img_arrays))
