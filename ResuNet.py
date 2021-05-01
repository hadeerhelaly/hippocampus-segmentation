import os
import sys
import random

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
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
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

## Seeding 
seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed


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
            
X_train = np.zeros((len(train_data), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
Y_train = np.zeros((len(train_data), IMG_HEIGHT, IMG_WIDTH, 1))



for file_index in range(len(train_data)):
    img = imread(train_data[file_index]) 
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH,IMG_CHANNELS), mode='constant', preserve_range=True)
    #img = denoise_bilateral(img, sigma_spatial=10,multichannel=True)
    #imshow(img)
    #plt.show()
    img = img/255.0
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
    mask = mask/255.0
    Y_train[n] = mask




dataset_path = "C:/Users/hadee/Desktop/data seg/"
train_path = os.path.join(dataset_path, "C:/Users/hadee/Desktop/data seg/")

class DataGen(keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, image_size=128):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()
        
    def __load__(self, id_name):
        ## Path
        image_path = os.path.join(self.path, "image", id_name) + ".jpg"
        mask_path = os.path.join(self.path, "masks", id_name) + ".jpg"
        
        ## Reading Image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        ##Reading Mask
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (self.image_size, self.image_size))
        mask = np.expand_dims(mask, axis=-1)
        
        ## Normalizaing 
        image = image/255.0
        mask = mask/255.0
        
        return image, mask
    
    def __getitem__(self, index):
        if(index+1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index*self.batch_size
        
        files_batch = self.ids[index*self.batch_size : (index+1)*self.batch_size]
        
        image = []
        mask  = []
        
        for id_name in files_batch:
            _img, _mask = self.__load__(id_name)
            image.append(_img)
            mask.append(_mask)
            
        image = np.array(image)
        mask  = np.array(mask)
        
        return image, mask
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.ids)/float(self.batch_size)))
    

train_csv = pd.read_csv(dataset_path + "train.csv")
train_ids = train_csv["id"].values

image_size = 128
batch_size = 16

val_data_size = 9

valid_ids = train_ids[:val_data_size]
train_ids = train_ids[val_data_size:]


gen = DataGen(train_ids, train_path, batch_size=batch_size, image_size=image_size)
x, y = gen.__getitem__(0)
print(x.shape, y.shape)

r = random.randint(0, len(X_train)-1)

fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
ax = fig.add_subplot(1, 2, 1)
ax.imshow(X_train[r])
ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(Y_train[r]*255, (image_size, image_size)), cmap="gray")


def bn_act(x, act=True):
    x = keras.layers.BatchNormalization()(x)
    if act == True:
        x = keras.layers.Activation("relu")(x)
    return x

def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    
    shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = keras.layers.Add()([conv, shortcut])
    return output

def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)
    
    shortcut = keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)
    
    output = keras.layers.Add()([shortcut, res])
    return output

def upsample_concat_block(x, xskip):
    u = keras.layers.UpSampling2D((2, 2))(x)
    c = keras.layers.Concatenate()([u, xskip])
    return c


def ResUNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, 3))
    
    ## Encoder
    e0 = inputs
    e1 = stem(e0, f[0])
    e2 = residual_block(e1, f[1], strides=2)
    e3 = residual_block(e2, f[2], strides=2)
    e4 = residual_block(e3, f[3], strides=2)
    e5 = residual_block(e4, f[4], strides=2)
    
    ## Bridge
    b0 = conv_block(e5, f[4], strides=1)
    b1 = conv_block(b0, f[4], strides=1)
    
    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, f[4])
    
    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, f[3])
    
    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, f[2])
    
    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, f[1])
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
    model = keras.models.Model(inputs, outputs)
    
    return model


smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = tf.layers.flatten(y_true)
    y_pred_f = tf.layers.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)



model = ResUNet()
adam = keras.optimizers.Adam()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=adam, loss=dice_coef_loss, metrics=[dice_coef])
model.summary()

results = model.fit(X_train, Y_train, validation_split=0.1,shuffle=True, batch_size=16, epochs=10)



train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
valid_gen = DataGen(valid_ids, train_path, image_size=image_size, batch_size=batch_size)

train_steps = len(train_ids)//batch_size
valid_steps = len(valid_ids)//batch_size

epochs = 10

model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, 
                    epochs=epochs)


model.save_weights("ResUNet.h5")

print("\n      Ground Truth            Predicted Value")

for i in range(1, 5, 1):
    ## Dataset for prediction
    x, y = valid_gen.__getitem__(i)
    result = model.predict(x)
    result = result > 0.4
    
    for i in range(len(result)):
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)

        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(np.reshape(y[i]*255, (image_size, image_size)), cmap="gray")

        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(np.reshape(result[i]*255, (image_size, image_size)), cmap="gray")
