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


def init_grad_cam(input_model, layer_name, actor=True):

    action = K.placeholder(shape=(), dtype=np.int32)
    if(actor):
        print("output[1]",input_model.output[1].shape)
        y_c = input_model.output[1][0, action]
    else:
        print("output[0]",input_model.output[0].shape)
        y_c = input_model.output[0][0, action]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    # Normalize if necessary
    #grads = normalize(grads)
    if (actor):
        gradient_function = K.function([input_model.input[0], action], [conv_output, grads])
    else:
        gradient_function = K.function([input_model.input[0],action], [conv_output, grads])
    return gradient_function

def grad_cam(gradient_function, frame, action, actor=True):
    """GradCAM method for visualizing input saliency."""
    
    if(actor):
        output, grads_val = gradient_function([frame,action])#,[action])
    else:
        output, grads_val = gradient_function([frame,0])
    
    weights = np.mean(grads_val, axis=(2,3))

    #print("out",output.shape)
    weights = weights[0,timestep,:]
    output = output[0,timestep,:,:,:]

    #weights = np.expand_dims(weights, axis=0)
    #cam = np.dot(output, weights)
    cam = np.zeros((7,7))
    for i in range(weights.shape[0]):
        cam += weights[i] * output[ :, : , i]


    #print(cam_.shape)
    cam = cv2.resize(cam, (84, 84), cv2.INTER_LINEAR)
    
    #cam = np.maximum(cam, 0)
    cam_max = cam.max() 
    if cam_max != 0: 
        cam = cam / cam_max
    cam[cam<0]=0
    #print(cam)
    return cam

