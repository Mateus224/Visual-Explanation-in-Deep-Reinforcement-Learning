from keras.models import Model
import keras
from keras import backend as K
from keras.optimizers import (Adam, RMSprop)
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
        Permute, merge, Merge,  Lambda, Reshape, TimeDistributed, LSTM, RepeatVector, Permute) #multiply,
from keras.layers.wrappers import Bidirectional
from keras.layers.wrappers import Bidirectional




def build_network(input_shape, num_actions):
    input_data = Input(shape = input_shape, name = "input")

    
    print('>>>> Defining Recurrent Modules...')
    input_data_expanded = Reshape((input_shape[0], input_shape[1], input_shape[2], 1), input_shape = input_shape) (input_data)
    #input_data_TimeDistributed = Permute((3, 1, 2, 4), input_shape=input_shape)(input_data_expanded)

    
    h1 = TimeDistributed(Convolution2D(32, 8, 8, subsample=(4, 4), activation = "relu"), \
        input_shape=(10, input_shape[0], input_shape[1], 1))(input_data_expanded)
    h2 = TimeDistributed(Convolution2D(64, 4, 4, subsample=(2, 2), activation = "relu"))(h1)
    h3 = TimeDistributed(Convolution2D(64, 3, 3, subsample=(1, 1), activation = "relu"))(h2)
    flatten_hidden = TimeDistributed(Flatten())(h3)
    hidden_input = TimeDistributed(Dense(512, activation = 'relu', name = 'flat_to_512')) (flatten_hidden)
    

    #Bidrection for a_fc(s,a) and v_fc layer
    ##################################
    if 1==1:#args.bidir:
        value_hidden =Bidirectional(LSTM(512, return_sequences=True, name = 'value_hidden', stateful=False, input_shape=(10, 512)), merge_mode='sum') (hidden_input) #Dense(512, activation = 'relu', name = 'value_fc')(all_outs)
        value_hidden_out = Bidirectional(LSTM(512, return_sequences=True, name = 'action_hidden_out', stateful=False, input_shape=(10, 512)), merge_mode='sum') (value_hidden)
        action_hidden =Bidirectional(LSTM(512, return_sequences=True,name = 'action_hidden', stateful=False, input_shape=(10, 512)), merge_mode='sum') (hidden_input) #Dense(512, activation = 'relu', name = 'value_fc')(all_outs)
        action_hidden_out = Bidirectional(LSTM(512, return_sequences=True,  name = 'action_hidden_out', stateful=False, input_shape=(10, 512)), merge_mode='sum') (action_hidden)

    else:
         value_hidden_out = LSTM(512, return_sequences=True, stateful=False, input_shape=(10, 512)) (hidden_input)
         action_hidden_out = LSTM(512, return_sequences=True, stateful=False, input_shape=(10, 512)) (hidden_input)
    
    value = TimeDistributed(Dense(1, name = "value"))(value_hidden_out)
    action = TimeDistributed(Dense(num_actions, name = "action"))(action_hidden_out)
    
    attention_vs = TimeDistributed(Dense(1, activation='tanh'),name = "AVS")(value) 
    attention_vs = Flatten()(attention_vs)
    attention_vs = Activation('softmax')(attention_vs)
    attention_vs = RepeatVector(1)(attention_vs)
    attention_vs = Permute([2, 1])(attention_vs)
    sent_representation_vs = merge([value, attention_vs], mode='mul',name = "Attention V")

    attention_pol = TimeDistributed(Dense(1, activation='tanh'),name = "AAS")(action) 
    attention_pol = Flatten()(attention_pol)
    attention_pol = Activation('softmax')(attention_pol)
    attention_pol = RepeatVector(num_actions)(attention_pol)
    attention_pol = Permute([2, 1])(attention_pol)
    sent_representation_policy =merge([action, attention_pol], mode='mul',name = "Attention P")


    context_value = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(1,))(sent_representation_vs)
    value = Dense(1, activation='linear', name='value')(context_value)
    context_policy = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(num_actions,))(sent_representation_policy)
    con_policy =Dense(num_actions, activation='relu')(context_policy)
    policy = Dense(num_actions, activation='softmax', name='policy')(con_policy)


    value_network = Model(input=input_data, output=value)
    policy_network = Model(input=input_data, output=policy)

    adventage = Input(shape=(1,))
    train_network = Model(input=[input_data, adventage], output=[value, policy])
    print(train_network.summary())
    return value_network, policy_network, train_network, adventage
