# Import libraries

import numpy as np
import scipy.io as io
import os 
from keras.layers import Input, Conv2D, Reshape, Add, MaxPooling2D, UpSampling2D, Activation
from keras.models import Sequential, Model, load_model
from keras import regularizers
from keras import optimizers
from keras.layers.merge import Concatenate

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

# Training dataset
# data = io.loadmat('training_Dataset.mat')

X_train_tmp = np.array(data['Avg_1'])
Y_train_tmp = np.array(data['Avg_8'])

# Data Augmentation
ss, xx, yy, zz = X_train_tmp.shape
X_flip1 = np.fliplr(X_train_tmp)
Y_flip1 = np.fliplr(Y_train_tmp)
X_flip2 = np.zeros([ss,xx,yy,zz])
Y_flip2 = np.zeros([ss,xx,yy,zz])
for slice in range(ss):
X_flip2[slice,:,:,:] = np.fliplr(np.squeeze(X_train_tmp[slice,:,:,:]))
Y_flip2[slice,:,:,:] = np.fliplr(np.squeeze(Y_train_tmp[slice,:,:,:]))
X_flip3 = np.fliplr(X_flip2)
Y_flip3 = np.fliplr(Y_flip2)

# Composition of the training dataset
X_train = np.zeros((ss*4,xx,yy,zz))
Y_train = np.zeros((ss*4,xx,yy,zz))

X_train[:,:,:,:] = np.concatenate((X_train_tmp,X_flip1,X_flip2,X_flip3),axis=0)
Y_train[:,:,:,:] = np.concatenate((Y_train_tmp,Y_flip1,Y_flip2,Y_flip3),axis=0)

# Network
input_shape = (192,192,2)
model_input = Input(shape=input_shape)

conv1 = Conv2D(64, (3, 3), 
           use_bias=False, padding="same",activation="relu",
           strides=1,kernel_initializer='glorot_uniform',
           name='block1_conv1')(model_input)
conv1 = Conv2D(64, (3, 3), 
           use_bias=False, padding="same",activation="relu",
           strides=1,kernel_initializer='glorot_uniform',
           name='block1_conv2')(conv1)

maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv1)

conv2 = Conv2D(128, (3, 3), 
           use_bias=False, padding="same",activation="relu",
           strides=1,kernel_initializer='glorot_uniform',
           name='block2_conv1')(maxpool1)
conv2 = Conv2D(128, (3, 3), 
           use_bias=False, padding="same",activation="relu",
           strides=1,kernel_initializer='glorot_uniform',
           name='block2_conv2')(conv2)

maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv2)

conv3 = Conv2D(256, (3, 3), 
           use_bias=False, padding="same",activation="relu",
           strides=1,kernel_initializer='glorot_uniform',
           name='block3_conv1')(maxpool2)
conv3 = Conv2D(256, (3, 3), 
           use_bias=False, padding="same",activation="relu",
           strides=1,kernel_initializer='glorot_uniform',
           name='block3_conv2')(conv3)

maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv3)

conv4 = Conv2D(512, (3, 3), 
           use_bias=False, padding="same",activation="relu",
           strides=1,kernel_initializer='glorot_uniform',
           name='block4_conv1')(maxpool3)
conv4 = Conv2D(512, (3, 3), 
           use_bias=False, padding="same",activation="relu",
           strides=1,kernel_initializer='glorot_uniform',
           name='block4_conv2')(conv4)

maxpool4 = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv4)

conv5 = Conv2D(1024, (3, 3), 
           use_bias=False, padding="same",activation="relu",
           strides=1,kernel_initializer='glorot_uniform',
           name='block5_conv1')(maxpool4)
conv5 = Conv2D(512, (3, 3), 
           use_bias=False, padding="same",activation="relu",
           strides=1,kernel_initializer='glorot_uniform',
           name='block5_conv2')(conv5)

upsampling1 = UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(conv5)

merge1 = Concatenate()([conv4,upsampling1])

conv6 = Conv2D(512, (3, 3), 
           use_bias=False, padding="same",activation="relu",
           strides=1,kernel_initializer='glorot_uniform',
           name='block6_conv1')(merge1)
conv6 = Conv2D(256, (3, 3), 
           use_bias=False, padding="same",activation="relu",
           strides=1,kernel_initializer='glorot_uniform',
           name='block6_conv2')(conv6)    

upsampling2 = UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(conv6)

merge2 = Concatenate()([conv3,upsampling2])

conv7 = Conv2D(256, (3, 3), 
           use_bias=False, padding="same",activation="relu",
           strides=1,kernel_initializer='glorot_uniform',
           name='block7_conv1')(merge2)
conv7 = Conv2D(128, (3, 3), 
           use_bias=False, padding="same",activation="relu",
           strides=1,kernel_initializer='glorot_uniform',
           name='block7_conv2')(conv7)

upsampling2 = UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(conv7)

merge3 = Concatenate()([conv2,upsampling2])

conv8 = Conv2D(128, (3, 3), 
           use_bias=False, padding="same",activation="relu",
           strides=1,kernel_initializer='glorot_uniform',
           name='block8_conv1')(merge3)
conv8 = Conv2D(64, (3, 3), 
           use_bias=False, padding="same",activation="relu",
           strides=1,kernel_initializer='glorot_uniform',
           name='block8_conv2')(conv8)

upsampling2 = UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')(conv8)

merge4 = Concatenate()([conv1,upsampling2])

conv9 = Conv2D(64, (3, 3), 
           use_bias=False, padding="same",activation="relu",
           strides=1,kernel_initializer='glorot_uniform',
           name='block9_conv1')(merge4)
conv9 = Conv2D(64, (3, 3), 
           use_bias=False, padding="same",activation="relu",
           strides=1,kernel_initializer='glorot_uniform',
           name='block9_conv2')(conv9)

model_output = Conv2D(2, (1, 1), 
           use_bias=False, padding="same",activation="linear",
           strides=1,kernel_initializer='glorot_uniform',
           name='block9_conv3')(conv9)
model = Model(model_input, model_output)
model.summary()

# Parameter setting and Training
from keras.optimizers import *
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))
history = model.fit(X_train, Y_train, batch_size=10, epochs=150,verbose=1)

# Save model weights
model.save('./Phase_denoising_Weight.h5')
