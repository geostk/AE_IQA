from keras import backend as K

import os,sys
from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
plt.switch_backend('agg')

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
from numpy import linalg
import skimage.util
import scipy.ndimage.filters
import scipy.misc
import scipy.io
from scipy import stats

import utils

import keras.backend.tensorflow_backend as KTF
KTF.set_session(utils.get_session())

################### Processing the data ##################
from keras.datasets import cifar10
(x_train, _), (x_test, _) = cifar10.load_data()

x_train = x_train.astype('float32')/255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))

x_test = x_test.astype('float32') / 255.
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))

# Adding GBLUR
x_train, x_train_noisy = utils.cifar10_gblur(x_train)
x_test, x_test_noisy = utils.cifar10_gblur(x_test)

x_train.shape, x_train_noisy.shape, x_test.shape, x_test_noisy.shape

# Displaying noisy images
n = 13
plt.figure(figsize=(20, 4))
for i in range(1, n):
# display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_train[i].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(x_train_noisy[i].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig('saves/cdA_gblur_noisy_vis.png')

"""Constructing the Model"""
import keras
import tensorflow as tf
import keras.backend as K
from keras import losses
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, add, Lambda

L = 5
F = [16, 32, 64, 128, 256] # Number of filters at each layer

# Clean Encoder
enc_inp = Input(shape = (32, 32, 3))
encs = []
enc = enc_inp
for i in range(L):
    enc = Conv2D(F[i], (3, 3), activation='relu', padding='same', strides=(2, 2))(enc)
    enc = Conv2D(F[i], (3, 3), activation='relu', padding='same', strides=(1, 1))(enc)
    encs.append(enc)

encoder = Model(inputs=[enc_inp], outputs=encs)
encoder.compile(optimizer='adadelta', loss='mean_squared_error')

# Quality
inp = Input(shape = (32, 32, 3))

enc = inp
for i in range(L):
    enc = Conv2D(F[i], (3, 3), activation='relu', padding='same', strides=(2, 2))(enc)
    enc = Conv2D(F[i], (3, 3), activation='relu', padding='same', strides=(1, 1))(enc)
    
    if i == L-1:
        Q = Dense(1, activation='relu')(enc)

quality = Model(inputs=[inp], outputs=[Q])
quality.compile(optimizer='adadelta', loss='mean_squared_error')

# Decoder
clean_acvns = []
for i in range(L):
    clean_acvn = Input(shape = (None, None, F[i]))
    clean_acvns.append(clean_acvn)

clean_dec = clean_acvns[L-1]
for i in range(-1, L-1)[::-1]:
    u = UpSampling2D((2, 2))(clean_dec)
    if i != -1:
        clean_dec = Conv2D(F[i], (3, 3), activation='relu', padding='same')(u)
        clean_dec = Conv2D(F[i], (3, 3), activation='relu', padding='same')(u)
        clean_dec = add([clean_dec, clean_acvns[i]])
    else:
        recon = Conv2D(3, (3, 3), activation='relu', padding='same')(u)

decoder = Model(inputs=clean_acvns, outputs=[recon])
decoder.compile(optimizer='adadelta', loss='mean_squared_error')

# Fitting clean and distorted images
clean_input = Input(shape=(32, 32, 3))
dist_input = Input(shape=(32, 32, 3))

encs = encoder(clean_input)
q = quality(dist_input)

corr_encs = encs[0 : L-1]
corr_enc = Lambda(lambda x: x[0]*x[1])([encs[L-1], q])
corr_encs.append(corr_enc)
dist_pred = decoder(corr_encs)

IQA_model = Model(inputs=[clean_input, dist_input], outputs=[dist_pred])
IQA_model.compile(optimizer='adadelta', loss=utils.loss_SSIM)

######################### Model Flow Diagram ###########################
#from keras.utils import plot_model
#plot_model(IQA_model, to_file='my_models/cdA_gblur_model.png', show_shapes=True)

########################## Training the model ##########################
from keras.callbacks import TensorBoard
import sys
IQA_model.fit([x_train, x_train_noisy],
              [x_train_noisy],
              verbose=2,
              epochs=100,
              batch_size=256,
              shuffle=True,
              validation_data=([x_test, x_test_noisy], 
                               [x_test_noisy]),
              callbacks=[TensorBoard(log_dir='tmp/IQA_model', histogram_freq=0, 
                                     write_graph=True, write_images=True)])

from keras.models import load_model
IQA_model.save('my_models/cdA_gblur.h5')

######################### Testing the model ############################
filt_imgs = IQA_model.predict([x_test[0 : 200], x_test_noisy[0 : 200]])

vis_clean = utils.visualize(x_test, [32, 32], [1, 1], [10, 10], color=1, channels='last')
vis_dist = utils.visualize(x_test_noisy, [32, 32], [1, 1], [10, 10], color=1, channels='last')
vis_filt = utils.visualize(filt_imgs, [32, 32], [1, 1], [10, 10], color=1, channels='last')

plt.figure(figsize=(25, 25))
plt.subplot(1, 3, 1), plt.imshow(vis_clean)
plt.subplot(1, 3, 2), plt.imshow(vis_dist)
plt.subplot(1, 3, 3), plt.imshow(vis_filt)
plt.show()
plt.savefig('saves/cdA_gblur_recon_vis.png')