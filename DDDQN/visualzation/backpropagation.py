import sys
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras import backend as K
from keras.preprocessing import image
from keras import backend as K
import visualisation

import tensorflow as tf
from tensorflow.python.framework import ops




def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """



def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (K.sqrt(K.mean(K.square(x))) + 1e-10)


def build_guided_model(actions):
    """Function returning modified model.
    
    Changes gradient function for all ReLu activations
    according to Guided Backpropagation.
    """
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                   tf.cast(op.inputs[0] > 0., dtype)

    g = K.get_session().graph
    with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
        return visualisation.build_network(actions)


def guided_backprop( guided_model, layer_name, frame):
    """Guided Backpropagation method for visualizing input saliency."""
    input_imgs = guided_model.input
    layer_output = guided_model.get_layer(layer_name).output
    #print(frame.shape, input_imgs, layer_output, "imput_images")
    grads = K.gradients(layer_output, input_imgs)[0]
    backprop_fn = K.function([input_imgs, K.learning_phase()], [grads])
    grads_val = backprop_fn([frame, 0])[0]
    return grads_val



    


def compute_saliency(model, guided_model, frame,action, layer_name='conv2d_6'):#, cls=-1, visualize=True, save=True):

    gb = guided_backpropa(frame, guided_model, layer_name)
 
        
    return gb#gradcam, gb, guided_gradcam


