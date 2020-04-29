import os
from keras import utils as np_utils
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pprint
import numpy as np
import glob
import keras
import cv2
from numpy.random import randint
#############################################################
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
from numpy import load
from matplotlib import pyplot
from numpy import expand_dims


# load an image to the preferred size
def load_image(filename, size=(192,192)):
    # load and resize the image
    pixels = load_img(filename, target_size=size)
    pixels, thresh1 = cv2.threshold(pixels,127,255,cv2.THRESH_BINARY)
    
    # convert to numpy array
    pixels = img_to_array(pixels)
    
    # transform in a sample
    pixels = expand_dims(pixels, 0)
    
    # scale from [0,255] to [-1,1]
    pixels = (pixels - 127.5) / 127.5
    return pixels

# load all images in a directory into memory
def load_images(path, size=(192,192)):
    data_list = list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path):
        
        if '.png' in filename or '.jpg' in filename:
            # load and resize the image
            pixels = load_img(path + filename, target_size=size)
            # convert to numpy array
            pixels = img_to_array(pixels)
            data_list.append(pixels)
    return asarray(data_list)

def compress_dataset(path, dataset_name):

    #Loading images from domain A
    dataA1 = load_images(path + 'trainA/')
    dataAB = load_images(path + 'testA/')
    dataA = vstack((dataA1, dataAB))
    print('Loaded dataA: ', dataA.shape)
    
    #Loading images from domain B
    dataB1 = load_images(path + 'trainB/')
    dataB2 = load_images(path + 'testB/')
    dataB = vstack((dataB1, dataB2))
    print('Loaded dataB: ', dataB.shape)
    
    # save as compressed numpy array
    filename = dataset_name+'.npz'
    savez_compressed(filename, dataA, dataB)
    print('Saved dataset: ', filename)
    
def plot_images_from_domains(n_samples, dataA, dataB):
    
    # plot source images

    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(dataA[i].astype('uint8'))
    # plot target image
    for i in range(n_samples):
        pyplot.subplot(2, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(dataB[i].astype('uint8'))
    pyplot.show()
    
def load_real_samples(filename):
    # load the dataset
    data = load(filename)
    
    # unpack arrays
    X1, X2 = data['arr_0'], data['arr_1']
    
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]

def save_models(step, g_model_AtoB, g_model_BtoA):
    
    # save the first generator model
    filename1 = 'g_model_AtoB_%06d.h5' % (step+1)
    g_model_AtoB.save(filename1)
    # save the second generator model
    filename2 = 'g_model_BtoA_%06d.h5' % (step+1)
    g_model_BtoA.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))
    

# select a random sample of images from the dataset
def select_sample(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    return X

def show_plot(imagesX, imagesY1, imagesY2):
    images = vstack((imagesX, imagesY1, imagesY2))
    titles = ['Real', 'Generated', 'Reconstructed']
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0
    # plot images row by row
    for i in range(len(images)):
        # define subplot
        pyplot.subplot(1, len(images), 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(images[i])
        # title
        pyplot.title(titles[i])
    pyplot.show()
 