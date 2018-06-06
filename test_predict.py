from unet import *
from data import *

mydata = dataProcess(512,512)

imgs_test = mydata.load_test_data()



myunet = myUnet()

model = myunet.get_unet()

# this function will load the data, predict on it and save as npy as well as jpgs
def predict():
    model.load_weights('unet.hdf5')
    imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
    if not os.path.exists('results'):
        os.makedirs('results')
        print("Created path " + str('results'))
    np.save('results/imgs_mask_test.npy', imgs_mask_test)

    print("array to image")
    imgs = np.load('results/imgs_mask_test.npy')
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = array_to_img(img)
        img.save("results/%d.jpg"%(i))

predict()
