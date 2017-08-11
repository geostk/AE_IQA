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
    plt.imshow(x_train[i+3*13*150].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(x_train_noisy[i+3*13*150].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig('saves/cdA_gblur_noisy_vis.png')

"""Constructing the Model"""
import keras
from keras import losses
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, add

L = 5
F = [32, 64, 128, 128, 256] # Number of filters at each layer
# Clean Encoder
clean_enc_inp = Input(shape = (None, None, 3))
encs = []
enc = clean_enc_inp
for i in range(L):
    enc = Conv2D(F[i], (3, 3), activation='relu', padding='same', strides=(2, 2))(enc) 
    encs.append(enc)

clean_encoder = Model(inputs=[clean_enc_inp], outputs=encs)
clean_encoder.compile(optimizer='adadelta', loss='mean_squared_error')

# Decoder
dec_inps = []
for i in range(L):
    dec_inp = Input(shape = (None, None, F[i]))
    dec_inps.append(dec_inp)
dec = dec_inps[L-1]
for i in range(-1, L-1)[::-1]:
    u = UpSampling2D((2, 2))(dec)
    if i != -1:
        dec = Conv2D(F[i], (3, 3), activation='relu', padding='same')(u)
        dec = add([dec, dec_inps[i]])
    else:
        recon = Conv2D(3, (3, 3), activation='relu', padding='same')(u)

decoder = Model(inputs=dec_inps, outputs=[recon])
decoder.compile(optimizer='adadelta', loss='mean_squared_error')

# Distortion Model
dist_enc_inp = Input(shape = (None, None, 3))
dist = dist_enc_inp
for i in range(L):
    dist = Conv2D(F[i], (3, 3), activation='relu', padding='same', strides=(2, 2))(dist)

dist_encoder = Model(inputs=[dist_enc_inp], outputs=[dist])
dist_encoder.compile(optimizer='adadelta', loss='mean_squared_error')

# Fitting clean images
clean_input = Input(shape = (None, None, 3))
clean_pred = decoder(clean_encoder(clean_input))

# Fitting distorted images
dist_input = Input(shape = (None, None, 3))
clean_encs = clean_encoder(clean_input)
dist_enc = dist_encoder(dist_input)
corr_enc = add([clean_encs[L-1], dist_enc])
dist_pred = decoder(clean_encs[:L-1] + [corr_enc])

IQA_model = Model(inputs=[clean_input, dist_input], outputs=[clean_pred, dist_pred])
IQA_model.compile(optimizer='adadelta', loss='mean_squared_error')

######################### Model Flow Diagram ###########################
#from keras.utils import plot_model
#plot_model(IQA_model, to_file='my_models/cdA_gblur_model.png', show_shapes=True)

########################## Training the model ##########################
from keras.callbacks import TensorBoard
import sys
IQA_model.fit([x_train, x_train_noisy],
              [x_train, x_train_noisy],
              epochs=100,
              batch_size=128,
              shuffle=True,
              validation_data=([x_test, x_test_noisy], 
                               [x_test, x_test_noisy]),
              callbacks=[TensorBoard(log_dir='tmp/IQA_model', histogram_freq=0, 
                                     write_graph=True, write_images=True)])

from keras.models import load_model
IQA_model.save('my_models/cdA_gblur.h5')

######################### Testing the model ############################
decoded_imgs = IQA_model.predict([x_test, x_test_noisy])

vis_clean = utils.visualize(x_test, [32, 32], [1, 1], [5, 5], color=1)
vis_dist = utils.visualize(x_test_noisy, [32, 32], [1, 1], [5, 5], color=1)
vis_clean_filt = utils.visualize(decoded_imgs[0], [32, 32], [1, 1], [5, 5], color=1)
vis_dist_filt = utils.visualize(decoded_imgs[1], [32, 32], [1, 1], [5, 5], color=1)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1), plt.imshow(vis_clean)
plt.subplot(2, 2, 2), plt.imshow(vis_dist)
plt.subplot(2, 2, 3), plt.imshow(vis_clean_filt)
plt.subplot(2, 2, 4), plt.imshow(vis_dist_filt)
plt.show()

plt.savefig('saves/cdA_gblur_recon_vis.png')
