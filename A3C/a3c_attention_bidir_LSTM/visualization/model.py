from keras.models import Model
from keras.layers import Input, Convolution2D, Flatten, Dense

def build_network(input_shape, output_shape):
    input_data = Input(shape = input_shape, name = "input")
    h = Convolution2D(32,8, 8, subsample=(4, 4), activation='relu')(input_data)
    h = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu')(h)
    h = Convolution2D(64, 3, 3, subsample=(2, 2), activation='relu')(h)
    h = Flatten()(h)
    h = Dense(256, activation='relu')(h)

    value = Dense(1, activation='linear')(h)
    policy = Dense(output_shape, activation='softmax')(h)

    value_network = Model(input=input_data, output=value)
    policy_network = Model(input=input_data, output=policy)

    adventage = Input(shape=(1,))
    train_network = Model(input=[input_data,adventage], output=[value, policy])
    print(train_network.summary())

    return value_network, policy_network, train_network, adventage
