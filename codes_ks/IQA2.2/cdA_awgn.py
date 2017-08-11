import os
import utils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

from keras.datasets import cifar10
(x_train, _), (x_test, _) = cifar10.load_data()

import keras.backend.tensorflow_backend as KTF
KTF.set_session(utils.get_session())
################### Processing the data ##################
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))  # adapt this if using `channels_first` image data format

# Adding AWGN
x_train, x_train_noisy = utils.cifar10_awgn(x_train)
x_test, x_test_noisy = utils.cifar10_awgn(x_test)
x_train.shape, x_train_noisy.shape, x_test.shape, x_test_noisy.shape

# Displaying noisy images
n = 10
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
plt.savefig('saves/cdA_awgn_noisy_vis.png')

""" Constructing the Model """
############## Autoencoder for reconstructing the clean image ###############
import keras
dist_input = Input(shape=(32, 32, 3))     # distorted image

############# Autoencoder for reconstructing the clean image ###############
x = Conv2D(32, (3, 3), activation='relu', padding='same')(dist_input)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
clean_inp_enc = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 32)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(clean_inp_enc)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
clean_recon = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

############# Autoencoder for reconstructing the distorted image ############
x = Conv2D(32, (3, 3), activation='relu', padding='same')(dist_input)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
dist_enc = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 32)
# Adding distortion to the encoded clean input
dist_inp_enc = keras.layers.add([clean_inp_enc, dist_enc])
x = Conv2D(32, (3, 3), activation='relu', padding='same')(dist_inp_enc)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
dist_recon = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(inputs=[dist_input], outputs=[clean_recon, dist_recon])
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

################## Model Flow Diagram ####################
from keras.utils import plot_model
plot_model(autoencoder, to_file='my_models/cdA_awgn_model.png')

########################## Training the model ##########################
from keras.callbacks import TensorBoard
import sys

autoencoder.fit([x_train_noisy],
                [x_train, x_train_noisy],
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=([x_test_noisy], [x_test, x_test_noisy]),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder', histogram_freq=0, write_graph=False)])

from keras.models import load_model
autoencoder.save('my_models/cdA_awgn.h5')

######################### Testing the model ############################
decoded_imgs = autoencoder.predict([x_test_noisy])

n = 20
plt.figure(figsize=(20, 4))
for i in range(1, n):
    # display original
    ax = plt.subplot(4, n, i)
    plt.imshow(x_test[i].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display distorted
    ax = plt.subplot(4, n, i + n)
    plt.imshow(x_test_noisy[i].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display original image reconstruction
    ax = plt.subplot(4, n, i + 2*n)
    plt.imshow(decoded_imgs[0][i].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display distorted image reconstruction
    ax = plt.subplot(4, n, i + 3*n)
    plt.imshow(decoded_imgs[1][i].reshape(32, 32, 3))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig('saves/cdA_awgn_recon_vis.png')