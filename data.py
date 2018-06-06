from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
import shutil
from scipy.misc import toimage, imsave
def show_image(image):
    toimage(image).show()

#import cv2
#from libtiff import TIFF

class myAugmentation(object):
    """
	A class used to augment images
	Firstly, read train image and label seperately, and then merge them together for the next process
	Secondly, use keras preprocessing to augmentate image
	Finally, seperate augmentated image apart into train image and label
	"""
    def __init__(self, train_path="data/train/image", label_path="data/train/label", merge_path="data/merge", aug_merge_path="data/aug_merge", aug_train_path="data/aug_train", aug_label_path="data/aug_label", img_type="tif"):
        """
		Using glob to get all .img_type from path
		"""
        self.train_imgs = glob.glob(train_path+"/*."+img_type)
        print("Fetched " + str(len(self.train_imgs)) + " train images")
        self.label_imgs = glob.glob(label_path+"/*."+img_type)
        print("Fetched " + str(len(self.label_imgs)) + " label images")
        self.train_path = train_path
        self.label_path = label_path
        self.merge_path = merge_path
        self.img_type = img_type
        self.aug_merge_path = aug_merge_path
        self.aug_train_path = aug_train_path
        self.aug_label_path = aug_label_path
        self.slices = len(self.train_imgs)
        self.datagen = ImageDataGenerator(
							        rotation_range=0.2,
							        width_shift_range=0.05,
							        height_shift_range=0.05,
							        shear_range=0.05,
							        zoom_range=0.05,
							        horizontal_flip=True,
							        fill_mode='nearest')

    def Augmentation(self):
        """
		Start augmentation.....
		"""
        trains = self.train_imgs
        labels = self.label_imgs
        path_train = self.train_path
        path_label = self.label_path
        path_merge = self.merge_path
        imgtype = self.img_type
        path_aug_merge = self.aug_merge_path
        if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
            print ("trains can't match labels")
            return 0
        for i in range(len(trains)):
            img_t = load_img(path_train+"/"+str(i)+"."+imgtype)
            img_l = load_img(path_label+"/"+str(i)+"."+imgtype)
            x_t = img_to_array(img_t)
            x_l = img_to_array(img_l)
            x_t[:,:,2] = x_l[:,:,0]
            img_tmp = array_to_img(x_t)
            # check if path exists before trying to save to it
            if not os.path.exists(path_merge):
                os.makedirs(path_merge)
                print("Created path " + str(path_merge))
            #else:
                #shutil.rmtree(path_merge)
                #print("Cleared " + str(path_merge))

            img_tmp.save(path_merge+"/"+str(i)+"."+imgtype)
            img = x_t
            img = img.reshape((1,) + img.shape)
            if not os.path.exists(path_aug_merge):
                os.makedirs(path_aug_merge)
                print("Created path " + str(path_aug_merge))
            #else:
                #shutil.rmtree(path_aug_merge)
                #print("Cleared " + str(path_aug_merge))
            savedir = path_aug_merge + "/" + str(i)
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            self.doAugmentate(img, savedir, str(i))


    def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='tif', imgnum=30):

        """
        augmentate one image
        """
        datagen = self.datagen
        i = 0
        for batch in datagen.flow(img,
                          batch_size=batch_size,
                          save_to_dir=save_to_dir,
                          save_prefix=save_prefix,
                          save_format=save_format):
            i += 1
            if i > imgnum:
                break

    def splitMerge(self):

        """
        split merged image apart
        """
        path_merge = self.aug_merge_path
        path_train = self.aug_train_path
        path_label = self.aug_label_path
        iter = 0
        if not os.path.exists(path_train):
            os.makedirs(path_train)
            print("Created path " + str(path_train))
        if not os.path.exists(path_label):
            os.makedirs(path_label)
            print("Created path " + str(path_label))
        for i in range(self.slices):
            path = path_merge + "/" + str(i)
            train_imgs = glob.glob(path+"/*."+self.img_type)

            #savedir = path_train + "/" + str(i)
            #if not os.path.exists(savedir):
                #os.makedirs(savedir)
                #print("Created path " + str(savedir))
            #else:
                #shutil.rmtree(savedir)
                #print("Cleared " + str(savedir))
            #if not os.path.lexists(savedir):
                #os.mkdir(savedir)

            #savedir = path_label + "/" + str(i)
            #if not os.path.lexists(savedir):
                #os.mkdir(savedir)
            #if not os.path.exists(savedir):
            #    print("Created path " + str(savedir))

            #for imgname in train_imgs:
            #    midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
            #    img = cv2.imread(imgname)
            #    img_train = img[:,:,2]#cv2 read image rgb->bgr
            #    img_label = img[:,:,0]
            #    cv2.imwrite(path_train+"/"+str(i)+"/"+midname+"_train"+"."+self.img_type,img_train)
            #    cv2.imwrite(path_label+"/"+str(i)+"/"+midname+"_label"+"."+self.img_type,img_label)

            for imgname in train_imgs:
                midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
                img = cv2.imread(imgname)
                img_train = img[:,:,2]#cv2 read image rgb->bgr
                img_label = img[:,:,0]
                cv2.imwrite(path_train+"/"+str(iter)+ "." +self.img_type,img_train)
                cv2.imwrite(path_label+"/"+str(iter)+ "." +self.img_type,img_label)
                iter = iter + 1

    def splitTransform(self):

        """
        split perspective transform images
        """
		#path_merge = "transform"
		#path_train = "transform/data/"
		#path_label = "transform/label/"
        path_merge = "deform/deform_norm2"
        path_train = "deform/train/"
        path_label = "deform/label/"
        train_imgs = glob.glob(path_merge+"/*."+self.img_type)
        for imgname in train_imgs:
            midname = imgname[imgname.rindex("/")+1:imgname.rindex("."+self.img_type)]
            img = cv2.imread(imgname)
            img_train = img[:,:,2]#cv2 read image rgb->bgr
            img_label = img[:,:,0]
            cv2.imwrite(path_train+midname+"."+self.img_type,img_train)
            cv2.imwrite(path_label+midname+"."+self.img_type,img_label)



class dataProcess(object):

    def __init__(self, out_rows, out_cols, data_path = "data/aug_train", label_path = "data/aug_label", test_path = "data/test", npy_path = "data/npydata", img_type = "tif"):
	#def __init__(self, out_rows, out_cols, data_path = "data/train/image", label_path = "data/train/label", test_path = "data/test", npy_path = "data/npydata", img_type = "tif"):


        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path

    def create_train_data(self):
        i = 0
        print('-'*30)
        print('Creating training images...')
        print('-'*30)

        imgs = glob.glob(self.data_path+"/*."+self.img_type)
        #imgs = glob.glob("data/train/image/*."+self.img_type)
        #LOOKHERE This is the part we changed so we could actually find some images
        print("Number of images found", len(imgs))
        imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
        for imgname in imgs:
            midname = imgname[imgname.rindex("/")+1:]
            #imgname is actually the name of the image
            img = load_img(self.data_path + "/" + midname,grayscale = True)
            label = load_img(self.label_path + "/" + midname,grayscale = True)
            print(midname)
            if i ==0:
                print(type(img))

            img = img_to_array(img)
            label = img_to_array(label)
            if i ==0:
                print((img.size))

			#img = cv2.imread(self.data_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#label = cv2.imread(self.label_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#img = np.array([img])
			#label = np.array([label])
            imgdatas[i] = img
            imglabels[i] = label
			#if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1
            print('loading done')

		#if not os.path.isfile(fname):
        if not os.path.exists(self.npy_path ):
            os.makedirs(self.npy_path)
            print("Created path " + str(self.npy_path))

        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        print(imgdatas.size)
        print(self.npy_path + '/imgs_train.npy')

        print('Saving to .npy files done.')

    def create_test_data(self):
        i = 0
        print('-'*30)
        print('Creating test images...')
        print('-'*30)
        imgs = glob.glob(self.test_path+"/*."+self.img_type)
        print("We have " + str(len(imgs)) + " test images")
        imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
        for imgname in imgs:
            midname = imgname[imgname.rindex("/")+1:]
            img = load_img(self.test_path + "/" + midname,grayscale = True)
            img = img_to_array(img)
			#img = cv2.imread(self.test_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#img = np.array([img])
            imgdatas[i] = img
            i += 1
        print('loading done')
        np.save('data/npydata/imgs_test.npy', imgdatas)
        print('Saving to imgs_test.npy files done.')

    def load_train_data(self):
        print('-'*30)
        print('load train images...')
        print('-'*30)
        imgs_train = np.load(self.npy_path+"/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
		#mean = imgs_train.mean(axis = 0)
		#imgs_train -= mean
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        return imgs_train,imgs_mask_train

    def load_test_data(self):
        print('-'*30)
        print('load test images...')
        print('-'*30)
        imgs_test = np.load(self.npy_path+"/imgs_test.npy")
        print(self.npy_path+"/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
		#mean = imgs_test.mean(axis = 0)
		#imgs_test -= mean
        return imgs_test


def performAug():
    #first we try to perform data augmentation, just to see what happens here
    aug = myAugmentation()
    aug.Augmentation()
    aug.splitMerge()
    aug.splitTransform()

def create_npys():
    print("first we initialize an object of the process class")
    mydata = dataProcess(512,512)
    print("now we create our train data npy files")
    mydata.create_train_data()
    print("now we create our npy files but for the test data")
    mydata.create_test_data()
    imgs_train,imgs_mask_train = mydata.load_train_data()
    #print (imgs_train.shape,imgs_mask_train.shape)
    print("CHECKING TO SEE HOW MANY THINGIES WE NOW HAVE")
    datapath = "data/npydata/imgs_train.npy"
    img_arrays = np.load(datapath)
    print(len(img_arrays))


if __name__ == "__main__":
#This function will fetch the data resulting from transformations and input
# This first function will peform data transformations
    #performAug()
# everything into a single npy file (one for train/label and test/label)
    #create_npys()
    import tensorflow as tf
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print(sess)
