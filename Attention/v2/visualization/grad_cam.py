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

def grad_cam(input_model, layer_name, frame, action):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, action]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]
    # Normalize if necessary
    #grads = normalize(grads)
    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([frame])
    print(output.shape,grads_val.shape)
    #output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val)#, axis=(1, 2))
    print(weights)
    #cam = np.dot(output, weights)

    cam = np.zeros(output.shape[1:])
    print(cam.shape)
    for i in range(weights.shape[0]):
        cam += weights[i] * output[i, :, :]

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam_max = cam.max() 
    if cam_max != 0: 
        cam = cam / cam_max
    return cam