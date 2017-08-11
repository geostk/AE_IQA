# Importing required libraries
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
import math
import scipy.io

########################## GRAPH ####################################
def autoencoder(dims=[32*32*3, 2000, 1000, 500]):
    """Build a deep denoising autoencoder w/ tied weights.
    Parameters
    ----------
    dimensions : list, optional
        The number of neurons for each layer of the autoencoder.
    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    """
    # input to the network
    with tf.device('/gpu:0'):
        X = tf.placeholder(tf.float32, [None, dims[0]], name='X')
        tilda_X = tf.placeholder(tf.float32, [None, dims[0]], name='tilda_X')

    # Build the encoder
    encoder = []
    current_input = tilda_X
    for layer_i, n_output in enumerate(dims[1:]):
        n_input = int(current_input.get_shape()[1])
        with tf.device('/gpu:0'):
            W = tf.Variable(
                tf.random_uniform([n_input, n_output],
                                  -1.0 / math.sqrt(n_input),
                                  1.0 / math.sqrt(n_input)))
            b = tf.Variable(tf.zeros([n_output]))
            output = tf.nn.tanh(tf.matmul(current_input, W) + b)            
        encoder.append(W)
        current_input = output
    # latent representation
    z = current_input
    encoder.reverse()
    # Build the decoder using the same weights
    for layer_i, n_output in enumerate(dims[:-1][::-1]):
        with tf.device('/gpu:0'):
            W = tf.transpose(encoder[layer_i])
            b = tf.Variable(tf.zeros([n_output]))
            output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        current_input = output
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    with tf.device('/gpu:0'):
        cost = tf.sqrt(tf.reduce_mean(tf.square(y - X)))
    return {'x': X, 'z': z, 'y': y,
            'cost': cost}    

def test_cifar10():
    import tensorflow as tf
    import numpy as np
    import matplotlib.pyplot as plt
    import utils
    import scipy.io        
    
# Loading the dataset
    from keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Obtaining reference and corrupted inputs
    x_train_clean, x_train_dist = utils.cifar10_awgn(x_train)
    x_test_clean, x_test_dist = utils.cifar10_awgn(x_test)
# Creating an instance of autoencoder
    ae = autoencoder(dims=[32*32*3, 2000, 1000, 500])
# Parameters
    training_epochs = 20
    batch_size = 256
    learning_rate = 0.000001
    display_step = 1
    examples_to_show = 15
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])
# We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
# Fit all training data
    no_batches = x_train_clean.shape[0]//batch_size
    for epoch in range(training_epochs):
        for batch in range(no_batches):
            batch_x, batch_tilda_x = utils.cifar10_next_batch(batch_size, 
                                                              x_train_clean, 
                                                              x_train_dist)
            _, c = sess.run([optimizer, ae['cost']], 
                            feed_dict={ae['X']: batch_x,
                                       ae['tilda_X']: batch_tilda_x})
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1),
              "cost=", "{:.9f}".format(c))

    recon = sess.run(ae['y'], feed_dict={ae['X']: x_test_clean[:examples_to_show], 
                                         ae['tilda_X']: x_test_dist[:examples_to_show]})
    f, a = plt.subplots(3, n_examples, figsize=(examples_to_show, 3))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(x_test_clean[i], (32, 32, 3)))
        a[1][i].imshow(np.reshape(x_test_dist[i], (32, 32, 3)))
        a[2][i].imshow(np.reshape(recon[i], (32, 32, 3)))
    plt.draw()
    plt.savefig('cifar10_saves/awgn_test_vis.png')
    plt.show()

if __name__ == '__main__':
    test_cifar10()