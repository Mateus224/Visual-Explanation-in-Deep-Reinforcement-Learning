import sys
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

import tensorflow as tf
from tensorflow.python.framework import ops

timestep=9


def init_grad_cam(input_model, layer_name):

    action = K.placeholder(shape=(), dtype=np.int32)
    print( input_model.output.shape)
    y_c = input_model.output[0, action]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    # Normalize if necessary
    #grads = normalize(grads)
    gradient_function = K.function([input_model.input, action], [conv_output, grads])

    return gradient_function

def grad_cam(gradient_function, frame, action):
    """GradCAM method for visualizing input saliency."""
    

    output, grads_val = gradient_function([frame,action])#,[action])
    #print("out",output)
    print("grads",grads_val)
    print("grads_val.shape:",grads_val.shape)
    weights = np.mean(grads_val, axis=(2,3))

    weights = weights[0,timestep,:]
    output = output[0,timestep,:,:,:]
    
    #weights = np.expand_dims(weights, axis=0)
    #cam = np.dot(output, weights)
    cam = np.zeros((20,20))
    for i in range(weights.shape[0]):
        cam += weights[i] * output[ :, : , i]


    #print(cam_.shape)
    cam = cv2.resize(cam, (84, 84), cv2.INTER_LINEAR)
    
    #cam = np.maximum(cam, 0)
    cam_max = cam.max() 
    if cam_max != 0: 
        cam = cam / cam_max
    
    return cam

