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
from keras.models import load_model
from keras_contrib import InstanceNormalization

video_path = 'videos/user28669847.avi'

if __name__ == "__main__":
   
    # loading the models
    cust = {'InstanceNormalization': InstanceNormalization}
    model_AtoB = load_model('g_model_AtoB_000965.h5', cust) #RGB->NIR
    model_BtoA = load_model('g_model_BtoA_000965.h5', cust) #NIR > RGB (does not work very well)

    #loading a video
    vidcap = cv2.VideoCapture(video_path)
    
    #getting first 100 frames of the video
    success,image = vidcap.read()
        
    frame_counter = 0
    while success and frame_counter < 200:
        success,image = vidcap.read()
        
        #pyplot.imshow(image)
        #pyplot.show()
        if success:
            #resizing frame  
            image = cv2.resize(image,(192,192))
            image = np.reshape(image,(1,192,192,3)) #model gets a list of images as input
            
            #predicting with RGB->NIR model
            image_tar = model_AtoB.predict(image) #RGB->NIR
            
            #scaling image
            image_tar = (image_tar + 1) / 2.0
            
            #showing image -> not using a horizontalstack cause it saturates RGB images when placed together with 0~1 images
            cv2.namedWindow('frames_NIR')       
            cv2.moveWindow('frames_NIR', 250, 50) #change this number to avoid overlap 
            
            #just moving one of the windows
            cv2.imshow('frames_RGB', image[0])
            cv2.imshow('frames_NIR', image_tar[0])
            
            cv2.normalize(image_tar[0], image_tar[0], 0, 255, cv2.NORM_MINMAX)
            #cv2.imwrite('videos/test.png', image_tar[0])
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            
            frame_counter += 1

    