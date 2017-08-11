import tensorflow as tf
import numpy as np

import scipy as sp
import scipy.ndimage.filters
import scipy.misc
import skimage.util
import scipy.io

from PIL import Image, ImageFile, ImageFilter
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os,sys
from os import listdir
from os.path import isfile, join
import random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def add_awgn(x, noise_fac):
    """
    Adds White Noise to each image given in the data matrix x
    """
    no_samples = x.shape[0]
    tilda_x = np.zeros(x.shape)
    
    for sam in range(no_samples):
        I = x[sam]
        tilda_I = I + noise_fac * np.random.randn(*I.shape)
        tilda_x[sam] = tilda_I
    return tilda_x

def add_gblur(x, r):
    """
    Adds Gaussian Blur to each image given in the data matrix x
    """
    no_samples = x.shape[0]
    tilda_x = np.zeros(x.shape)

    for sam in range(no_samples):
        im = Image.fromarray(np.uint8(x[sam] * 255.))
        tilda_I = np.array(im.filter(ImageFilter.GaussianBlur(radius=r)))
        tilda_x[sam] = tilda_I/255.
    return tilda_x

def add_jpeg(x, quality):
    """
    Adds JPEG artifacts to each image given in the data matrix x
    """
    no_samples = x.shape[0]
    tilda_x = np.zeros(x.shape)
    
    for sam in range(no_samples):
        arr = x[sam].reshape(x.shape[1 : ])
        scipy.misc.imsave('inp.jpg', arr)
        I = Image.open('inp.jpg')
        I.save('out.jpg', format='JPEG', quality=quality)
        tilda_I = np.array(Image.open('out.jpg'))
        tilda_x[sam] = tilda_I.astype('float32')/255.
        os.remove("inp.jpg")
        os.remove("out.jpg")
    return tilda_x

def cifar10_awgn(x):
    """
    This function adds AWGN to the images 'x' given
    """
    num_samples = x.shape[0]
# Initializing clean and distorted samples
    x_clean = np.zeros((5*num_samples,)+x.shape[1:])
    x_dist = np.zeros((5*num_samples,)+x.shape[1:])
# Processing the samples
    for idx in range(num_samples):
        I_clean = x[idx]
    # No distortion
        I_dist = I_clean
        x_clean[5*idx] = I_clean
        x_dist[5*idx] = I_dist
    # Distortion level 1
        I_dist = I_clean + 0.001 * np.random.randn(*I_clean.shape)
        x_clean[5*idx+1] = I_clean
        x_dist[5*idx+1] = I_dist
    # Distortion level 2
        I_dist = I_clean + 0.01 * np.random.randn(*I_clean.shape)
        x_clean[5*idx+2] = I_clean
        x_dist[5*idx+2] = I_dist
    # Distortion level 3
        I_dist = I_clean + 0.1 * np.random.randn(*I_clean.shape)
        x_clean[5*idx+3] = I_clean
        x_dist[5*idx+3] = I_dist
    # Distortion level 4
        I_dist = I_clean + 0.5 * np.random.randn(*I_clean.shape)
        x_clean[5*idx+4] = I_clean
        x_dist[5*idx+4] = I_dist
    # Display
        if (idx+1)%5000 == 0:
            print '%d/%d images done!' % (idx+1, num_samples)
    print 'Adding AWGN done!'
    return x_clean, x_dist

def cifar10_gblur(x):
    """
    This function adds GBLUR to the images 'x' given
    """
    radii = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
    num_samples = x.shape[0]
    L = 13   # Number of levels of distortion
# Initializing clean and distorted samples
    x_clean = np.zeros((L*num_samples,)+x.shape[1:])
    x_dist = np.zeros((L*num_samples,)+x.shape[1:])
# Processing the samples
    for idx in range(num_samples):
        I_clean = x[idx]
        im = Image.fromarray(np.uint8(I_clean * 255.))
    # No distortion
        I_dist = I_clean
        x_clean[L*idx] = I_clean
        x_dist[L*idx] = I_dist

        for i in range(1, L):
        # Distortion level i
            I_dist = np.array(im.filter(ImageFilter.GaussianBlur(radius=radii[i])))
            x_clean[L*idx+i] = I_clean
            x_dist[L*idx+i] = I_dist/255.
    # Display
        if (idx+1)%5000 == 0:
            print '%d/%d images done!' % (idx+1, num_samples)
    print 'Adding GBLUR done!'
    return x_clean, x_dist
                      
def cifar10_jpeg(x):
    """
    This function adds JPEG artifcats to the images 'x' given
    """
    num_samples = x.shape[0]
# Initializing clean and distorted samples
    x_clean = np.zeros((5*num_samples,)+x.shape[1:]) 
    x_dist = np.zeros((5*num_samples,)+x.shape[1:])
# Processing the samples
    for idx in range(num_samples):
        I_clean = x[idx].reshape(x.shape[1:])
        scipy.misc.imsave('inp.jpg', I_clean)
        Ibin = Image.open('inp.jpg')        
    # No distortion
        I_dist = I_clean
        x_clean[5*idx] = I_clean
        x_dist[5*idx] = I_dist
    # Distortion level 1
        Ibin.save('out.jpg', format='JPEG', quality=75)
        I_dist = np.array(Image.open('out.jpg'))
        x_clean[5*idx+1] = I_clean
        x_dist[5*idx+1] = I_dist.astype('float32')/255.
    # Distortion level 2
        Ibin.save('out.jpg', format='JPEG', quality=50)
        I_dist = np.array(Image.open('out.jpg'))
        x_clean[5*idx+2] = I_clean
        x_dist[5*idx+2] = I_dist.astype('float32')/255.
    # Distortion level 3
        Ibin.save('out.jpg', format='JPEG', quality=25)
        I_dist = np.array(Image.open('out.jpg'))
        x_clean[5*idx+3] = I_clean
        x_dist[5*idx+3] = I_dist.astype('float32')/255.
    # Distortion level 4
        Ibin.save('out.jpg', format='JPEG', quality=10)
        I_dist = np.array(Image.open('out.jpg'))
        x_clean[5*idx+4] = I_clean
        x_dist[5*idx+4] = I_dist.astype('float32')/255.
        os.remove("inp.jpg")
        os.remove("out.jpg")
    # Display
        if (idx+1)%5000 == 0:
            print '%d/%d images done!' % (idx+1, num_samples)
    print 'Adding JPEG done!'
    return x_clean, x_dist

def visualize(x, filt_size, filt_spacing, vis_size, color, channels):
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
    channels: 'first' or 'last'
    """
    no_filts = x.shape[0]
    Fh, Fw = filt_size      # Single filter height and width
    Sr, Sc = filt_spacing   # Filter spacing in rows and columns
    Nr, Nc = vis_size       # No. of filters in rows and columns
    
    Vh = (Fh + Sr)*Nr - Sr  # Visualization height
    Vw = (Fw + Sc)*Nc - Sc  # Visualization width

    indcs = np.random.permutation(Nr * Nc) # Permutation of indices to choose randomly

    if color == 0:
    # Initialization of the visualization matrix
        vis_x = np.zeros((Vh, Vw))

        for num_row in range(Nr):
            for num_col in range(Nc):
                idx = Nc*num_row + num_col
                I = x[idx].reshape((Fh, Fw))
                vis_x[num_row * (Fh + Sr) : num_row * (Fh + Sr) + Fh,
                      num_col * (Fw + Sc) : num_col * (Fw + Sc) + Fw] = I
        return vis_x
    elif color == 1:
    # Initialization of the visualization matrix
        vis_x = np.zeros((Vh, Vw, 3))
        for num_row in range(Nr):
            for num_col in range(Nc):
#                idx = indcs[Nc*num_row + num_col]
                idx = Nc*num_row + num_col
                if channels == 'last':
                    I = x[idx].reshape((Fh, Fw, 3))
                elif channels == 'first':
                    I = np.transpose(x[idx].reshape((3, Fh, Fw)), [1, 2, 0])
                else:
                    print ("************* ERROR in channels *************")
                vis_x[num_row * (Fh + Sr) : num_row * (Fh + Sr) + Fh,
                      num_col * (Fw + Sc) : num_col * (Fw + Sc) + Fw, :] = I
        return vis_x
    else:
        print ("********* ERROR in color **********")

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

def get_session(gpu_fraction=0.25):
    '''Total GPU Memory: 12GB Allocated memory:~3GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#    gpu_options = tf.GPUOptions(allow_growth = True)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
def entropy_patch(patch):
    bin_counts, bin_edges = np.histogram(patch, bins=64)
    probs = bin_counts/float(np.sum(bin_counts))
    ent = np.sum([-p*np.log2(p) for p in probs if p!=0])
    return ent
    
def getPatchesDict(I, psize, noPatches, entropy_th, stride):
# Size of the input image
    M = I.shape[0] 
    N = I.shape[1]
# Size of the patch
    w = psize[0]
    h = psize[1]
# stride for patch selection
    Sx = stride[0]
    Sy = stride[1]
# Patch Locations (Left most and top most)
    x = [i for i in range(0, M - w + 1, Sx)]
    y = [i for i in range(0, N - h + 1, Sy)]
# Initialization
    dummy = np.zeros((1, w, h, 3))
    Patches = dummy
# Extracting the patches    
    for row in range(len(x)):
        for col in range(len(y)):
            px = x[row]
            py = y[col]
            patch = I[px : px + w, py : py + h, :]
            
#            if (entropy_patch(rgb2gray(patch)) > entropy_th):
            patch = patch.reshape((1,) + patch.shape)
            Patches = np.concatenate((Patches, patch), axis=0)

    Patches = Patches[1 :]    # Removing the dummy
                
# Randomly selecting the patches
    if (Patches.shape[0] > noPatches):
        indcs = random.sample(range(Patches.shape[0]), noPatches)
        Patches = Patches[indcs]
    return Patches/255.

def gblur_patches(Iclean):
    Pclean = getPatchesDict(Iclean, [32, 32], noPatches=4000, entropy_th=1.0, stride=[32, 32])
    idx = Pclean.shape[0]
    x_clean = np.zeros((13*idx, 32, 32, 3))
    x_dist  = np.zeros((13*idx, 32, 32, 3))
    im = Image.fromarray(np.uint8(Iclean))
# No distortion
    Pdist = Pclean
    x_clean[0 : idx, :, :, :] = Pclean
    x_dist[0 : idx, :, :, :]  = Pdist
# Distortion level 1
    Idist = np.array(im.filter(ImageFilter.GaussianBlur(radius=0.5)))
    Pdist = getPatchesDict(Idist, [32, 32], noPatches=4000, entropy_th=1.0, stride=[32, 32])
    x_clean[idx : 2*idx, :, :, :] = Pclean
    x_dist[idx : 2*idx, :, :, :]  = Pdist
# Distortion level 2
    Idist = np.array(im.filter(ImageFilter.GaussianBlur(radius=1)))
    Pdist = getPatchesDict(Idist, [32, 32], noPatches=4000, entropy_th=1.0, stride=[32, 32])    
    x_clean[2*idx : 3*idx, :, :, :] = Pclean
    x_dist[2*idx : 3*idx, :, :, :]  = Pdist
# Distortion level 3
    Idist = np.array(im.filter(ImageFilter.GaussianBlur(radius=1.5)))
    Pdist = getPatchesDict(Idist, [32, 32], noPatches=4000, entropy_th=1.0, stride=[32, 32])    
    x_clean[3*idx : 4*idx, :, :, :] = Pclean
    x_dist[3*idx : 4*idx, :, :, :]  = Pdist
# Distortion level 4
    Idist = np.array(im.filter(ImageFilter.GaussianBlur(radius=2)))
    Pdist = getPatchesDict(Idist, [32, 32], noPatches=4000, entropy_th=1.0, stride=[32, 32])
    x_clean[4*idx : 5*idx, :, :, :] = Pclean
    x_dist[4*idx : 5*idx, :, :, :]  = Pdist
# Distortion level 5
    Idist = np.array(im.filter(ImageFilter.GaussianBlur(radius=2.5)))
    Pdist = getPatchesDict(Idist, [32, 32], noPatches=4000, entropy_th=1.0, stride=[32, 32])
    x_clean[5*idx : 6*idx, :, :, :] = Pclean
    x_dist[5*idx : 6*idx, :, :, :]  = Pdist
# Distortion level 6
    Idist = np.array(im.filter(ImageFilter.GaussianBlur(radius=3)))
    Pdist = getPatchesDict(Idist, [32, 32], noPatches=4000, entropy_th=1.0, stride=[32, 32])
    x_clean[6*idx : 7*idx, :, :, :] = Pclean
    x_dist[6*idx : 7*idx, :, :, :]  = Pdist
# Distortion level 7
    Idist = np.array(im.filter(ImageFilter.GaussianBlur(radius=3.5)))
    Pdist = getPatchesDict(Idist, [32, 32], noPatches=4000, entropy_th=1.0, stride=[32, 32])
    x_clean[7*idx : 8*idx, :, :, :] = Pclean
    x_dist[7*idx : 8*idx, :, :, :]  = Pdist
# Distortion level 8
    Idist = np.array(im.filter(ImageFilter.GaussianBlur(radius=4)))
    Pdist = getPatchesDict(Idist, [32, 32], noPatches=4000, entropy_th=1.0, stride=[32, 32])
    x_clean[8*idx : 9*idx, :, :, :] = Pclean
    x_dist[8*idx : 9*idx, :, :, :]  = Pdist
# Distortion level 9
    Idist = np.array(im.filter(ImageFilter.GaussianBlur(radius=4.5)))
    Pdist = getPatchesDict(Idist, [32, 32], noPatches=4000, entropy_th=1.0, stride=[32, 32])
    x_clean[9*idx : 10*idx, :, :, :] = Pclean
    x_dist[9*idx : 10*idx, :, :, :]  = Pdist
# Distortion level 10
    Idist = np.array(im.filter(ImageFilter.GaussianBlur(radius=5)))
    Pdist = getPatchesDict(Idist, [32, 32], noPatches=4000, entropy_th=1.0, stride=[32, 32])
    x_clean[10*idx : 11*idx, :, :, :] = Pclean
    x_dist[10*idx : 11*idx, :, :, :]  = Pdist
# Distortion level 11
    Idist = np.array(im.filter(ImageFilter.GaussianBlur(radius=5.5)))
    Pdist = getPatchesDict(Idist, [32, 32], noPatches=4000, entropy_th=1.0, stride=[32, 32])
    x_clean[11*idx : 12*idx, :, :, :] = Pclean
    x_dist[11*idx : 12*idx, :, :, :]  = Pdist
# Distortion level 12
    Idist = np.array(im.filter(ImageFilter.GaussianBlur(radius=6)))
    Pdist = getPatchesDict(Idist, [32, 32], noPatches=4000, entropy_th=1.0, stride=[32, 32])
    x_clean[12*idx : 13*idx, :, :, :] = Pclean
    x_dist[12*idx : 13*idx, :, :, :]  = Pdist
    
    return x_clean, x_dist

def seg_gblur():
# Segmentation Dataset
#1
    datadir = '/data2/rajeev/dA_IQA/data/BSDS300/images/train/'
    large_files = [f for f in listdir(datadir) if isfile(join(datadir, f))]
    N = 13*150
    x_train = np.zeros((300*N, 32, 32, 3))
    x_train_noisy = np.zeros((300*N, 32, 32, 3))

    for i in range(len(large_files)):
        f = datadir + large_files[i]
        I = mpimg.imread(f)
        large_clean, large_noisy = gblur_patches(I)
        x_train[N*i : N*(i+1)] = large_clean
        x_train_noisy[N*i : N*(i+1)] = large_noisy
    # Display
        if (i+1)%50 == 0:
            string = str(i+1) + 'images done!'
            print string
#2
    datadir = '/data2/rajeev/dA_IQA/data/BSDS300/images/test/'
    large_files = [f for f in listdir(datadir) if isfile(join(datadir, f))]
    for i in range(len(large_files)):
        f = datadir + large_files[i]
        I = mpimg.imread(f)
        large_clean, large_noisy = gblur_patches(I)
        x_train[N*(i+200) : N*(i+201)] = large_clean
        x_train_noisy[N*(i+200) : N*(i+201)] = large_noisy
    # Display
        if (i+1)%50 == 0:
            string = str(i+1+200) + 'images done!'
            print string
    return x_train, x_train_noisy

import tensorflow as tf
import keras.backend as K

def loss_SSIM(y_true, y_pred):
    patches_true = tf.extract_image_patches(y_true, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
    patches_pred = tf.extract_image_patches(y_pred, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
    u_true = K.mean(patches_true, axis=-1)
    u_pred = K.mean(patches_pred, axis=-1)
    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)
    std_true = K.sqrt(var_true)
    std_pred = K.sqrt(var_pred)
    covar = K.mean(patches_true*patches_pred, axis=-1) - u_true*u_pred
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    num = (2 * u_true * u_pred + c1) * (2 * covar + c2)
    den = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    ssim = num/den
    ssim = tf.where(tf.is_nan(ssim), K.zeros_like(ssim), ssim)
    ssim = K.mean(ssim, (1, 2))
    
    return K.mean(-K.log(ssim))