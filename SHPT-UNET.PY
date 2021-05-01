
import tensorflow as tf
import os
import random
import numpy as np
 
from tqdm import tqdm 

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
from skimage.util.montage import montage2d
import tensorflow as tf
import numpy as np,sys,os
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
from skimage import img_as_float
#from skimage.metrics import peak_signal_noise_ratio
from matplotlib import pyplot as plt
from skimage import io
from scipy import ndimage as nd
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
from skimage import img_as_float
import cv2
from sklearn import metrics
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import itertools
import cv2 
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
import matplotlib.pyplot as plt

seed = 42
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
IMAGE_SIZE=(IMG_WIDTH,IMG_HEIGHT,IMG_CHANNELS)
        
data = "C:/Users/hadee/Desktop/seghipp0/images"
train_data = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data)):
    for filename in fileList:
        if ".jpg" in filename.lower():  # check whether the file's DICOM
            train_data.append(os.path.join(dirName,filename))
            

data_left = "C:/Users/hadee/Desktop/seghipp0/masks/left"
mask_left = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_left)):
    for filename in fileList:
        if ".jpg" in filename.lower():  # check whether the file's DICOM
            mask_left.append(os.path.join(dirName,filename))

data_right = "C:/Users/hadee/Desktop/seghipp0/masks/right"
mask_right = []  # create an empty list
for dirName, subdirList, fileList in sorted(os.walk(data_right)):
    for filename in fileList:
        if ".jpg" in filename.lower():  # check whether the file's DICOM
            mask_right.append(os.path.join(dirName,filename))

X_train = np.zeros((len(train_data), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_data), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)



for file_index in range(len(train_data)):
    img = imread(train_data[file_index]) 
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), mode='constant', preserve_range=True)
    #img = denoise_bilateral(img, sigma_spatial=10,multichannel=True)
    #imshow(img)
    #plt.show()
    X_train[file_index] = img 
    
for n in range(len(mask_right)):    
    maskl = imread(mask_left[n])
    maskr = imread(mask_right[n])

    
    mask = np.maximum(maskl, maskr)
    
    #mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH))
    #mask = denoise_bilateral(mask, sigma_spatial=10,multichannel=True)
    #print(n)
    
    mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH,1),mode='constant', preserve_range=True)
    
    
    #masksq= np.squeeze(mask)
    
    Y_train[n] = mask
    
    
import skimage.io                                     #Used for imshow function
import skimage.transform                              #Used for resize function
from skimage.morphology import label 

id = 10
print(X_train[id].shape)
skimage.io.imshow(X_train[id])
plt.show()
skimage.io.imshow(Y_train[id][:,:,0])
plt.show()


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix


image_x = random.randint(0, len(train_data))
imshow(X_train[image_x])
plt.show()
imshow(np.squeeze(Y_train[image_x]))
plt.show()




fig, ax = plt.subplots(1,3,figsize = (16,12))
ax[0].imshow(X_train[image_x], cmap = 'gray')

ax[1].imshow(np.squeeze(Y_train[image_x]), cmap = 'gray')

ax[2].imshow(X_train[image_x], cmap = 'gray', interpolation = 'none')
ax[2].imshow(np.squeeze(Y_train[image_x]), cmap = 'jet', interpolation = 'none', alpha = 0.7)






fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(montage2d(X_train[:, :, :, 0]))
ax1.set_title('MRI input images samples ')
ax2.imshow(montage2d(Y_train[:, :, :, 0]))
ax2.set_title('Ground Truth masks samples')



X_train,Y_train=shuffle(X_train,Y_train)
X_train, X_test, Y_train, y_test = train_test_split(X_train, Y_train, random_state=42, test_size=0.1)    

imshow(np.squeeze(y_test[0]))
plt.show()



smooth=1    


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

 
   
#Build the model
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)
 
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)
 
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)
 
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path 
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
 
u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
 
u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
 
u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
 
outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
 
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

################################
#Modelcheckpoint
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=False)

callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs')]

results = model.fit(X_train, Y_train, validation_split=0.1,shuffle=True, batch_size=16, epochs=10)

####################################

preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

y_test=y_test.astype(np.float)
#preds_test=preds_test.astype(np.uint8)
 
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)



id = random.randint(0, len(preds_test_t))
print(X_test[id].shape)
skimage.io.imshow(X_test[id])
skimage.io.imshow(np.squeeze(y_test[id]))
plt.show()
skimage.io.imshow(preds_test[id][:,:,0])
plt.show()

fig, ax = plt.subplots(1,4,figsize = (16,12))
ax[0].imshow(X_test[image_x], cmap = 'gray')

ax[1].imshow(np.squeeze(preds_test[image_x]), cmap = 'gray')
ax[2].imshow(np.squeeze(y_test[image_x]), cmap = 'gray')
ax[3].imshow(X_test[image_x], cmap = 'gray', interpolation = 'none')
ax[3].imshow(np.squeeze(preds_test[image_x]), cmap = 'jet', interpolation = 'none', alpha = 0.7)


plt.subplot(1,3,1)
plt.title("X_test")
plt.axis('off')
skimage.io.imshow(X_test[id])
plt.subplot(1,3,2)
plt.title("Y_test")
plt.axis('off')
skimage.io.imshow(np.squeeze(y_test[id]))
plt.subplot(1,3,3)
plt.title("Prediction")
plt.axis('off')
skimage.io.imshow(preds_test[id][:,:,0])
plt.show()






# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_test_t))
imshow(X_test[ix])
plt.show()
imshow(np.squeeze(y_test[ix]))
plt.show()

test_img = preds_test_t[:,:, 0]

plt.imshow(test_img)

imshow(np.squeeze(preds_test[2]))
plt.show()

# Perform a sanity check on some random validation samples
ix = random.randint(0, len(preds_val_t))
imshow(X_train[int(X_train.shape[0]*0.9):][ix])
plt.show()
imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix]))
plt.show()
imshow(np.squeeze(preds_test_t[ix]))
plt.show()

fig, ax = plt.subplots(1,3,figsize = (16,12))
ax[0].imshow(X_test[image_x], cmap = 'gray')

ax[1].imshow(np.squeeze(Y_test[image_x]), cmap = 'gray')

ax[2].imshow(X_test[image_x], cmap = 'gray', interpolation = 'none')
ax[2].imshow(np.squeeze(Y_test[image_x]), cmap = 'jet', interpolation = 'none', alpha = 0.7)






fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
ax1.imshow(montage2d(X_test[:, :, :, 0]))
ax1.set_title('MRI input images samples ')
ax2.imshow(montage2d(Y_test[:, :, :, 0]))
ax2.set_title('Ground Truth masks samples')



def plotPredictions(a,b,c,d,e):
    model = e
    # Threshold predictions
    preds_train = model.predict(a[:int(a.shape[0]*0.9)], verbose=1)
    preds_val = model.predict(a[int(a.shape[0]*0.9):], verbose=1)
    preds_test = model.predict(c, verbose=1)
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_val_t = (preds_val > 0.5).astype(np.uint8)
    preds_test = (preds_test > 0.5).astype(np.uint8)
    # Perform a sanity check on some random training samples
    ix = random.randint(0, len(preds_train_t))
    plt.subplot(1,3,1)
    plt.title("X_train")
    plt.axis('off')
    imshow(a[ix])
    plt.subplot(1,3,2)
    plt.title("Y_train")
    plt.axis('off')
    imshow(np.squeeze(b[ix]))
    plt.subplot(1,3,3)
    plt.title("Prediction")
    plt.axis('off')
    imshow(np.squeeze(preds_train_t[ix]))
    plt.show()
    # Perform a sanity check on some random validation samples
    ix = random.randint(0, len(preds_val_t))
    plt.subplot(1,3,1)
    plt.title("X_test")
    plt.axis('off')
    imshow(a[int(a.shape[0]*0.9):][ix])
    plt.subplot(1,3,2)
    plt.title("Y_test")
    plt.axis('off')
    imshow(np.squeeze(b[int(b.shape[0]*0.9):][ix]))
    plt.subplot(1,3,3)
    plt.title("Prediction")
    plt.axis('off')
    imshow(np.squeeze(preds_val_t[ix]))
    plt.show()
plotPredictions(X_train,Y_train,X_test,y_test, model)



#Graphing our training and validation
acc = results.history['acc']
val_acc = results.history['val_acc']
loss = results.history['loss']
val_loss = results.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy') 
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend()
plt.show()

model.summary()




