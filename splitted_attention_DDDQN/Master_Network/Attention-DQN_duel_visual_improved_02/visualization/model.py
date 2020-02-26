from keras.models import Model
import keras
from keras.optimizers import (Adam, RMSprop)
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
        Permute, merge, Merge,  Lambda, Reshape, TimeDistributed, LSTM, RepeatVector, Permute) #multiply,
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

import sys
from gym import wrappers
import tensorflow as tf
import numpy as np
from keras.utils.visualize_util import plot as plot_model




def build_network(input_shape, num_actions):
    num_frames=10
    input_data = Input(shape = input_shape, name = "input")
    print('>>>> Defining Recurrent Modules...')
    input_data_expanded = Reshape((input_shape[0], input_shape[1], input_shape[2], 1), input_shape = input_shape) (input_data)
    input_data_TimeDistributed = Permute((3, 1, 2, 4), input_shape=input_shape)(input_data_expanded)
    h1 = TimeDistributed(Convolution2D(32, 8, 8, subsample=(4, 4), activation = "relu", name = "conv1"), \
        input_shape=(num_frames, input_shape[0], input_shape[1], 1))(input_data_TimeDistributed)
    h2 = TimeDistributed(Convolution2D(64, 4, 4, subsample=(2, 2), activation = "relu", name = "conv2"))(h1)
    h3 = TimeDistributed(Convolution2D(64, 3, 3, subsample=(1, 1), activation = "relu", name = "conv3"))(h2)
    flatten_hidden = TimeDistributed(Flatten())(h3)
    hidden_input = TimeDistributed(Dense(512, activation = 'relu', name = 'flat_to_512')) (flatten_hidden)


    #Duel with splitted Attention in V(s)- und A(s)-Attention
    #if mode == "duel_at_improved":
    print("Improved duel Attention Network")
    #Bidrection for a_fc(s,a) and v_fc layer
    ##################################
#       if args.bidir:
    value_hidden =Bidirectional(LSTM(512, return_sequences=True,  name = 'value_hidden', stateful=False, input_shape=(num_frames, 512)), merge_mode='sum') (hidden_input) #Dense(512, activation = 'relu', name = 'value_fc')(all_outs)
    value_hidden_out = Bidirectional(LSTM(512, return_sequences=True,  name = 'action_hidden_out', stateful=False, input_shape=(num_frames, 512)), merge_mode='sum') (value_hidden)
    action_hidden =Bidirectional(LSTM(512, return_sequences=True,  name = 'action_hidden', stateful=False, input_shape=(num_frames, 512)), merge_mode='sum') (hidden_input) #Dense(512, activation = 'relu', name = 'value_fc')(all_outs)
    action_hidden_out = Bidirectional(LSTM(512, return_sequences=True,  name = 'action_hidden_out', stateful=False, input_shape=(num_frames, 512)), merge_mode='sum') (action_hidden)

#        else:
#             value_hidden_out = LSTM(512, return_sequences=True, stateful=False, input_shape=(args.num_frames, 512)) (hidden_input)
#             action_hidden_out = LSTM(512, return_sequences=True, stateful=False, input_shape=(args.num_frames, 512)) (hidden_input)
    
    value = TimeDistributed(Dense(1, name = "value"))(value_hidden_out)
    action = TimeDistributed(Dense(18, name = "action"))(action_hidden_out)
    


    attention_vs = TimeDistributed(Dense(1, activation='tanh'),name = "AVS")(value) 
    attention_vs = Flatten()(attention_vs)
    attention_vs = Activation('softmax')(attention_vs)
    attention_vs = RepeatVector(1)(attention_vs)
    attention_vs = Permute([2, 1])(attention_vs)
    sent_representation_vs = merge([value, attention_vs], mode='mul',name = "Attention V")

    attention_as = TimeDistributed(Dense(1, activation='tanh'),name = "AAS")(action) 
    attention_as = Flatten()(attention_as)
    attention_as = Activation('softmax')(attention_as)
    attention_as = RepeatVector(18)(attention_as)
    attention_as = Permute([2, 1])(attention_as)
    sent_representation_as =merge([action, attention_as], mode='mul',name = "Attention A")

    action_mean = Lambda(lambda x: tf.reduce_mean(x, axis = 2, keep_dims = True), name = 'action_mean')(sent_representation_as) 
    output = Lambda(lambda x: x[0] + (x[1] - x[2]), name = 'duel_output')([sent_representation_as, sent_representation_vs, action_mean])
  

    #Duel with one attention
    #elif mode == "duel_at":
    #    if args.bidir:
    #        hidden_input = Bidirectional(LSTM(512, return_sequences=True, stateful=False, input_shape=(args.num_frames, 512)), merge_mode='sum') (hidden_input)
    #        all_outs = Bidirectional(LSTM(512, return_sequences=True, stateful=False, input_shape=(args.num_frames, 512)), merge_mode='sum') (hidden_input)
    #    else:
    #        all_outs = LSTM(512, return_sequences=True, stateful=False, input_shape=(args.num_frames, 512)) (hidden_input)  
        
    #    value_hidden = Dense(512, activation = 'relu', name = 'value_fc')(all_outs)
    #    value = Dense(1, name = "value")(value_hidden)

    #    action_hidden = Dense(512, activation = 'relu', name = 'action_fc')(all_outs)
    #    action = Dense(num_actions, name = "action")(action_hidden)
    #    action_mean = Lambda(lambda x: tf.reduce_mean(x, axis = 1, keep_dims = True), name = 'action_mean')(action)
    #    output = Lambda(lambda x: x[0] + x[1] - x[2], name = 'duel_output')([action,value, action_mean])    
        # attention
    #    attention = TimeDistributed(Dense(1, activation='tanh'))(output)
    #    attention = Flatten()(attention)
    #    attention = Activation('softmax')(attention)
    #    attention = RepeatVector(18)(attention)
    #    attention = Permute([2, 1])(attention)
    #    output = merge([output, attention], mode='mul')                 

    context = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(18,))(output)

    output = Dense(num_actions, name = "output")(context)

    model = Model(input = input_data, output = output)
    print(model.summary())
    return model
