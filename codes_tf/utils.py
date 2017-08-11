import tensorflow as tf
import numpy as np

import scipy as sp
import scipy.ndimage.filters
import scipy.misc
import skimage.util
import scipy.io

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os,sys

def add_gblur(x):
    """
    Adds Gaussian Blur to each image given in the data matrix x
    """
    no_samples = x.shape[0]
    no_dim = x.shape[1]
    tilda_x = np.zeros(x.shape)

    for sam in range(no_samples):
        I = x[sam, :].reshape((28, 28))
        tilda_I = sp.ndimage.filters.gaussian_filter(I, sigma=5.0)
        tilda_x[sam, :] = tilda_I.reshape((1, no_dim))
    return tilda_x

def add_awgn(x):
    """
    Adds White Noise to each image given in the data matrix x
    """
    no_samples = x.shape[0]
    no_dim = x.shape[1]
    tilda_x = np.zeros(x.shape)
    
    for sam in range(no_samples):
        I = x[sam, :].reshape((28, 28))
        tilda_I = I + 0.2 * np.random.randn(*I.shape)
        tilda_x[sam, :] = tilda_I.reshape((1, no_dim))
    return tilda_x

def add_jpeg(x):
    """
    Adds JPEG artifacts to each image given in the data matrix x
    """
    no_samples = x.shape[0]
    no_dim = x.shape[1]
    tilda_x = np.zeros(x.shape)
    
    for sam in range(no_samples):
        arr = x[sam, :].reshape(28, 28)
        scipy.misc.imsave('inp.jpg', arr)
        I = Image.open('inp.jpg')
        I.save('out.jpg', format='JPEG', quality=10)
        tilda_I = np.array(Image.open('out.jpg'))
        tilda_x[sam, :] = tilda_I.reshape((1, no_dim))
        os.remove("inp.jpg")
        os.remove("out.jpg")
    return tilda_x

def visualize(x, filt_size, filt_spacing, vis_size, color):
    """
    This function is to visualize matrices for ex: weights learned
    in any learning problem
    x: x is a numpy array of the learned filters each arranged in rows 
       to be visualized
    filt_size: Size of each filter in x
    filt_spacing: Space between each filter in the visualization of x
    vis_size: Size of the final visualization to be produced
    color: '0' means x is a grayscale input
           '1' means x is a RGB input
    """
    no_filts = x.shape[0]
    Fh, Fw = filt_size      # Single filter height and width
    Sr, Sc = filt_spacing   # Filter spacing in rows and columns
    Nr, Nc = vis_size       # No. of filters in rows and columns
    
    Vh = (Fh + Sr)*Nr - Sr  # Visualization height
    Vw = (Fw + Sc)*Nc - Sc  # Visualization width

    if color == 0:
    # Initialization of the visualization matrix
        vis_x = np.zeros((Vh, Vw), dtype='uint8')

        for num_row in range(Nr):
            for num_col in range(Nc):
                idx = num_row * num_col
                I = x[idx, :].reshape((Fh, Fw))
                vis_x[num_row * (Fh + Sr) : num_row * (Fh + Sr) + Fh,
                      num_col * (Fw + Sc) : num_col * (Fw + Sc) + Fw] = I * 255
        return vis_x
    elif color == 1:
    # Initialization of the visualization matrix
        vis_x = np.zeros((Vh, Vw, 3), dtype='uint8')

        for num_row in range(Nr):
            for num_col in range(Nc):
                idx = num_row * num_col
                I = x[idx, :].reshape((Fh, Fw, 3))
                vis_x[num_row * (Fh + Sr) : num_row * (Fh + Sr) + Fh,
                      num_col * (Fw + Sc) : num_col * (Fw + Sc) + Fw, :] = I * 255
        return vis_x
    else:
        print ("*********ERROR**********")

def cifar10_awgn(x):
    """
    This function adds AWGN to the images 'x' given
    """
    num_samples = x.shape[0]
    # Initializing clean and distorted samples
    x_clean = np.zeros((5*num_samples, 32*32*3), dtype='uint8') 
    x_dist = np.zeros((5*num_samples, 32*32*3), dtype='uint8')

    # Processing the samples
    for idx in range(num_samples):
        arr = x[idx, :, :, :]
        # No distortion
        I_clean = arr.reshape(1, 32*32*3)
        I_dist = I_clean
        x_clean[5*idx, :] = I_clean
        x_dist[5*idx, :] = I_dist
        # Distortion level 1
        I_dist = 255*skimage.util.random_noise(arr, mode='gaussian', var=0.001).reshape(1,32*32*3)
        x_clean[5*idx+1, :] = I_clean
        x_dist[5*idx+1, :] = I_dist
        # Distortion level 2
        I_dist = 255*skimage.util.random_noise(arr, mode='gaussian', var=0.01).reshape(1,32*32*3)
        x_clean[5*idx+2, :] = I_clean
        x_dist[5*idx+2, :] = I_dist
        # Distortion level 3
        I_dist = 255*skimage.util.random_noise(arr, mode='gaussian', var=0.05).reshape(1,32*32*3)
        x_clean[5*idx+3, :] = I_clean
        x_dist[5*idx+3, :] = I_dist
        # Distortion level 4
        I_dist = 255*skimage.util.random_noise(arr, mode='gaussian', var=0.1).reshape(1,32*32*3)
        x_clean[5*idx+4, :] = I_clean
        x_dist[5*idx+4, :] = I_dist

        # Display
        if (idx+1)%5000 == 0:
            print '%d/%d images done!' % (idx+1, num_samples)
    print 'Adding AWGN done!'
    return x_clean, x_dist

def cifar10_gblur(x):
    """
    This function adds GBLUR to the images 'x' given
    """
    num_samples = x.shape[0]
    # Initializing clean and distorted samples
    x_clean = np.zeros((5*num_samples, 32*32*3), dtype='uint8') 
    x_dist = np.zeros((5*num_samples, 32*32*3), dtype='uint8')

    # Processing the samples
    for idx in range(num_samples):
        arr = x[idx, :, :, :]
        # No distortion
        I_clean = arr.reshape(1, 32*32*3)
        I_dist = I_clean
        x_clean[5*idx, :] = I_clean
        x_dist[5*idx, :] = I_dist
        # Distortion level 1
        I_dist = scipy.ndimage.filters.gaussian_filter(arr, sigma=0.5).reshape(1, 32*32*3)
        x_clean[5*idx+1, :] = I_clean
        x_dist[5*idx+1, :] = I_dist
        # Distortion level 2
        I_dist = scipy.ndimage.filters.gaussian_filter(arr, sigma=1.0).reshape(1, 32*32*3)
        x_clean[5*idx+2, :] = I_clean
        x_dist[5*idx+2, :] = I_dist
        # Distortion level 3
        I_dist = scipy.ndimage.filters.gaussian_filter(arr, sigma=1.2).reshape(1, 32*32*3)
        x_clean[5*idx+3, :] = I_clean
        x_dist[5*idx+3, :] = I_dist
        # Distortion level 4
        I_dist = scipy.ndimage.filters.gaussian_filter(arr, sigma=1.5).reshape(1, 32*32*3)
        x_clean[5*idx+4, :] = I_clean
        x_dist[5*idx+4, :] = I_dist

        # Display
        if idx%5000 == 0:
            print '%d/%d images done!' % (idx, num_samples)
    print 'Adding GBLUR done!'
    return x_clean, x_dist

def cifar10_jpeg(x):
    """
    This function adds JPEG to the images 'x' given
    """
    num_samples = x.shape[0]
    # Initializing clean and distorted samples
    x_clean = np.zeros((32, 32, 3, 5*num_samples), dtype='uint8') 
    x_dist = np.zeros((32, 32, 3, 5*num_samples), dtype='uint8')

    # Processing the samples
    for idx in range(num_samples):
        I_clean = x[idx, :, :, :]
        scipy.misc.imsave('inp.jpg', I_clean)
        Ibin = Image.open('inp.jpg')        
        # No distortion
        I_dist = I_clean
        x_clean[:, :, :, 5*idx] = I_clean
        x_dist[:, :, :, 5*idx] = I_dist
        # Distortion level 1
        Ibin.save('out.jpg', format='JPEG', quality=75)
        I_dist = np.array(Image.open('out.jpg'))
        x_clean[:, :, :, 5*idx+1] = I_clean
        x_dist[:, :, :, 5*idx+1] = I_dist
        # Distortion level 2
        Ibin.save('out.jpg', format='JPEG', quality=50)
        I_dist = np.array(Image.open('out.jpg'))
        x_clean[:, :, :, 5*idx+2] = I_clean
        x_dist[:, :, :, 5*idx+2] = I_dist
        # Distortion level 3
        Ibin.save('out.jpg', format='JPEG', quality=25)
        I_dist = np.array(Image.open('out.jpg'))
        x_clean[:, :, :, 5*idx+3] = I_clean
        x_dist[:, :, :, 5*idx+3] = I_dist
        # Distortion level 4
        Ibin.save('out.jpg', format='JPEG', quality=10)
        I_dist = np.array(Image.open('out.jpg'))
        x_clean[:, :, :, 5*idx+4] = I_clean
        x_dist[:, :, :, 5*idx+4] = I_dist
        os.remove("inp.jpg")
        os.remove("out.jpg")
        # Display
        if idx%5000 == 0:
            print '%d/%d images done!' % (idx, num_samples)
    print 'Adding JPEG done!'
    return x_clean, x_dist

def next_batch(num, data):
    """
    Return a total of `num` samples from the array `data`. 
    """
    idx = np.arange(0, len(data))  # get all possible indexes
    np.random.shuffle(idx)  # shuffle indexes
    idx = idx[0:num]  # use only `num` random indexes
    data_shuffle = [data[i] for i in idx]  # get list of `num` random samples
    data_shuffle = np.asarray(data_shuffle)  # get back numpy array

    return data_shuffle

def cifar10_next_batch(num, x, tilda_x):
    """
    Return a total of `num` samples from x (reference images) and 
    tilda_x (distorted images)
    """
    idx = np.arange(0, len(x))     # get all possible indexes
    np.random.shuffle(idx)         # shuffle indexes
    idx = idx[0:num]               # use only `num` random indexes
    batch_x = [x[i] for i in idx]  # get list of `num` random samples
    batch_x = np.asarray(batch_x)  # get back numpy array
    batch_tilda_x = [tilda_x[i] for i in idx]
    batch_tilda_x = np.asarray(batch_tilda_x)

    return batch_x, batch_tilda_x