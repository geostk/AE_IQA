import utils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K

from keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28))  # adapt this if using `channels_first` image data format

# Adding JPEG
x_train, x_train_noisy = utils.cifar10_jpeg(x_train)
x_test, x_test_noisy = utils.cifar10_jpeg(x_test)

x_train = x_train.reshape(x_train.shape + (1,))
x_train_noisy = x_train_noisy.reshape(x_train_noisy.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))
x_test_noisy = x_test_noisy.reshape(x_test_noisy.shape + (1,))

x_train.shape, x_train_noisy.shape, x_test.shape, x_test_noisy.shape

# Displaying noisy images
vis_clean = utils.visualize(x_test, [28, 28], [1, 1], [10, 10], color=0)
vis_dist = utils.visualize(x_test_noisy, [28, 28], [1, 1], [10, 10], color=0)

plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.imshow(vis_clean)
plt.title('Clean Images', fontsize='14', fontweight='bold')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(vis_dist)
plt.title('Distorted Images', fontsize='14', fontweight='bold')
plt.axis('off')
plt.show()
plt.savefig('saves/cdA_jpeg_noisy_vis.png')

# Building the Graph
input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 32)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.utils import plot_model
plot_model(autoencoder, to_file='my_models/cdA_jpeg_model.png')

from keras.callbacks import TensorBoard
import sys

autoencoder.fit(x_train_noisy, x_train,
                epochs=100,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder', histogram_freq=0, write_graph=False)])

from keras.models import load_model
autoencoder.save('my_models/cdA_jpeg.h5')

x_test_filt = autoencoder.predict(x_test_noisy)
vis_dist = utils.visualize(x_test_noisy, [28, 28], [1, 1], [10, 10], color=0)
vis_filt = utils.visualize(x_test_filt, [28, 28], [1, 1], [10, 10], color=0)

plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.imshow(vis_dist)
plt.title('Distorted Images', fontsize='14', fontweight='bold')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(vis_filt)
plt.title('Filtered Images', fontsize='14', fontweight='bold')
plt.axis('off')
plt.show()
plt.savefig('saves/cdA_jpeg_recon_vis.png')