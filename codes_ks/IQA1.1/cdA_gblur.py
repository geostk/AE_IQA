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
x_train, x_train_noisy = utils.cifar10_gblur(x_train)

x_test = x_test.astype('float32') / 255.
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))
x_test, x_test_noisy = utils.cifar10_gblur(x_test)

print x_train.shape, x_train_noisy.shape, x_test.shape, x_test_noisy.shape

# Displaying noisy images
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_train[i+4*13*150].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(x_train_noisy[i+4*13*150].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig('saves/cdA_gblur_noisy_vis.png')

"""Constructing the Model"""
import keras
from keras import losses
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, concatenate

L = 5  # Total number of layers
S = 2  # Number of shared layers
F = [16, 32, 64, 128, 256]     # Number of clean filters including input

# Clean different encoding layers
inp = Input(shape=(32, 32, 3))
enc = inp
for i in range(L-S):
    enc = Conv2D(F[i], (3, 3), activation='relu', padding='same', strides=(2, 2))(enc)

clean_diff_enc = Model(inputs=[inp], outputs=[enc])
clean_diff_enc.compile(optimizer='adadelta', loss='mean_squared_error')

# Distorted different encoding layers
inp = Input(shape=(32, 32, 3))
enc = inp
for i in range(L-S):
    enc = Conv2D(F[i], (3, 3), activation='relu', padding='same', strides=(2, 2))(enc)

dist_diff_enc = Model(inputs=[inp], outputs=[enc])
dist_diff_enc.compile(optimizer='adadelta', loss='mean_squared_error')

# Shared Encoding Layers
inp = Input(shape=(4, 4, F[L-S-1]))
enc = inp
for i in range(L-S, L):
    enc = Conv2D(F[i], (3, 3), activation='relu', padding='same', strides=(2, 2))(enc)

comm_enc = Model(inputs=[inp], outputs=[enc])
comm_enc.compile(optimizer='adadelta', loss='mean_squared_error')

# Shared Decoding Layers
inp = Input(shape=(1, 1, F[L-1]))
dec = inp
for i in range(L-S-1, L-1)[::-1]:
    dec = UpSampling2D((2, 2))(dec)
    dec = Conv2D(F[i], (3, 3), activation='relu', padding='same')(dec)
comm_dec = Model(inputs=[inp], outputs=[dec])
comm_dec.compile(optimizer='adadelta', loss='mean_squared_error')

# Clean different decoding layers
inp = Input(shape=(4, 4, F[L-S-1]))
dec = inp
for i in range(-1,L-S)[::-1]:
    if i == -1:
        recon = Conv2D(3, (3, 3), activation='relu', padding='same')(dec)
    else:
        dec = UpSampling2D((2, 2))(dec)
        dec = Conv2D(F[i], (3, 3), activation='relu', padding='same')(dec)
clean_diff_dec = Model(inputs=[inp], outputs=[recon])
clean_diff_dec.compile(optimizer='adadelta', loss='mean_squared_error')

# Distorted different decoding layers
inp = Input(shape=(4, 4, F[L-S-1]))
dec = inp
for i in range(-1,L-S)[::-1]:
    if i == -1:
        recon = Conv2D(3, (3, 3), activation='relu', padding='same')(dec)
    else:
        dec = UpSampling2D((2, 2))(dec)
        dec = Conv2D(F[i], (3, 3), activation='relu', padding='same')(dec)
dist_diff_dec = Model(inputs=[inp], outputs=[recon])
dist_diff_dec.compile(optimizer='adadelta', loss='mean_squared_error')

# clean input
clean_input = Input(shape = (32, 32, 3))
clean_pred = clean_diff_dec(comm_dec(comm_enc(clean_diff_enc(clean_input))))

# distorted Input
dist_input = Input(shape = (32, 32, 3))
dist_pred = dist_diff_dec(comm_dec(comm_enc(dist_diff_enc(dist_input))))

# IQA Model
IQA_model = Model(inputs=[clean_input, dist_input], outputs=[clean_pred, dist_pred])
IQA_model.compile(optimizer='adadelta', loss=[utils.loss_SSIM, utils.loss_SSIM])

######################### Model Flow Diagram ###########################
#from keras.utils import plot_model
#plot_model(IQA_model, to_file='my_models/cdA_gblur_model.png', show_shapes=True)

########################## Training the model ##########################
from keras.callbacks import TensorBoard
import sys
IQA_model.fit([x_train, x_train_noisy],
              [x_train, x_train_noisy],
              epochs=50,
              verbose = 2,
              batch_size=512,
              shuffle=True,
              validation_data=([x_test, x_test_noisy], 
                               [x_test, x_test_noisy]),
              callbacks=[TensorBoard(log_dir='tmp/IQA_model', histogram_freq=0, 
                                     write_graph=True, write_images=True)])

from keras.models import load_model
IQA_model.save('my_models/cdA_gblur.h5')

######################### Testing the model ############################
decoded_imgs = IQA_model.predict([x_test, x_test_noisy])

vis_clean = utils.visualize(x_test, [32, 32], [1, 1], [13, 13], color=1, channels='last')
vis_dist = utils.visualize(x_test_noisy, [32, 32], [1, 1], [13, 13], color=1, channels='last')
vis_clean_filt = utils.visualize(decoded_imgs[0], [32, 32], [1, 1], [13, 13], color=1, channels='last')
vis_dist_filt = utils.visualize(decoded_imgs[1], [32, 32], [1, 1], [13, 13], color=1, channels='last')

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1), plt.imshow(vis_clean)
plt.subplot(2, 2, 2), plt.imshow(vis_dist)
plt.subplot(2, 2, 3), plt.imshow(vis_clean_filt)
plt.subplot(2, 2, 4), plt.imshow(vis_dist_filt)
plt.show()

plt.savefig('saves/cdA_gblur_recon_vis.png')
