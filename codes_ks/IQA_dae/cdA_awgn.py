import os
import utils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from keras import backend as K
import keras.backend.tensorflow_backend as KTF
KTF.set_session(utils.get_session())

from keras.datasets import cifar10
(x_train, _), (x_test, _) = cifar10.load_data()

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
vis_clean = utils.visualize(x_train, [32, 32], [1, 1], [5, 5], color=1)
vis_dist = utils.visualize(x_train_noisy, [32, 32], [1, 1], [5, 5], color=1)

plt.figure(figsize=(10, 10))
plt.subplot(1,2,1)
plt.imshow(vis_clean)
plt.subplot(1,2,2)
plt.imshow(vis_dist)
plt.show()
plt.savefig('saves/cdA_awgn_noisy_vis.png')

""" Constructing the Model """
############## Autoencoder for reconstructing the clean image ###############
import keras
from keras import losses
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, concatenate

inp = Input(shape=(32, 32, 3))    # Input image
x = Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

h = MaxPooling2D((2, 2), padding='same')(x) # at this point the representation is (8, 8, 32)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(h)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)

recon = Conv2D(3, (3, 3), activation='relu', padding='same')(x)

autoencoder = Model(inputs=[inp], outputs=[recon])
autoencoder.compile(optimizer='adadelta', loss=['mean_squared_error'])

################## Model Flow Diagram ####################
from keras.utils import plot_model
plot_model(autoencoder, to_file='my_models/cdA_awgn_model.png')

########################## Training the model ##########################
from keras.callbacks import TensorBoard
import sys

autoencoder.fit([x_train_noisy],
                [x_train],
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=([x_test_noisy], [x_test]),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder', histogram_freq=0, write_graph=False)])

from keras.models import load_model
autoencoder.save('my_models/cdA_awgn.h5')

######################### Testing the model ############################
filt = autoencoder.predict([x_test_noisy])
vis_filt = utils.visualize(filt, [32, 32], [1, 1], [5, 5], color=1)
vis_dist = utils.visualize(x_test_noisy, [32, 32], [1, 1], [5, 5], color=1)

plt.figure(figsize=(10, 10))
plt.subplot(1,2,1)
plt.imshow(vis_dist)
plt.subplot(1,2,2)
plt.imshow(vis_filt)
plt.show()
plt.savefig('saves/cdA_awgn_recon_vis.png')
