import os
from keras import utils as np_utils
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pprint
import numpy as np
import glob
import keras
import cv2
from numpy import load
from matplotlib import pyplot
import utils as translate_utils
import models.models as model
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras_contrib import InstanceNormalization

#dataset path
path = 'C:/Users/igorc/Desktop/Skin detection dataset/'

if __name__ == "__main__":
    
    
    #if you need to compress the dataset, call utils.compress_dataset(path, dataset_name)
    translate_utils.compress_dataset(path, 'Skin_detection')
    
    '''
    #loading compressed dataset
    data = load('Maxtrack_Raw.npz')
    
    #getting images from both domains
    dataA, dataB = data['arr_0'], data['arr_1']
    print('Loaded: ', dataA.shape, dataB.shape)
    
    #plotting images from both domains
    translate_utils.plot_images_from_domains(3, dataA, dataB) #the first parameter is the number of images to be plotted
    
    # define input shape based on the loaded dataset
    dataset = translate_utils.load_real_samples('MaxTrack2.npz')
    image_shape = dataset[0].shape[1:]
    
    # generator: A -> B
    cust = {'InstanceNormalization': InstanceNormalization}
    g_model_AtoB = model.define_generator(image_shape)
    #g_model_AtoB = load_model('C:/Users/igorc/Desktop/Image_Translation/src/g_model_AtoB_008000.h5', cust)
    
    # generator: B -> A
    g_model_BtoA = model.define_generator(image_shape)
    #g_model_BtoA = load_model('C:/Users/igorc/Desktop/Image_Translation/src/g_model_BtoA_008000.h5', cust)
    
    # discriminator: A -> [real/fake]
    d_model_A = model.define_discriminator(image_shape)
    
    # discriminator: B -> [real/fake]
    d_model_B = model.define_discriminator(image_shape)
    
    # composite: A -> B -> [real/fake, A]
    c_model_AtoB = model.define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
    
    # composite: B -> A -> [real/fake, B]
    c_model_BtoA = model.define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
    
    # train models
    model.train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset)
    '''