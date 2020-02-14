from scipy.misc.pilutil import imresize
from scipy.misc.pilutil import imread
from skimage.color import rgb2gray
from multiprocessing import *
from collections import deque
import traceback
import gym
import numpy as np
import h5py
import argparse
from keras.models import Model
import keras
from keras import backend as K
from keras.optimizers import (Adam, RMSprop)
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
        Permute, merge, Merge,  Lambda, Reshape, TimeDistributed, LSTM, RepeatVector, Permute) #multiply,
from keras.layers.wrappers import Bidirectional
from keras.layers.wrappers import Bidirectional
from visualization.backpropagation import build_guided_model

from play_analyse import play_game


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


class ActingAgent(object):
    def __init__(self, action_space, screen=(84, 84)):
        self.screen = screen
        self.input_depth = 1
        self.past_range = 10
        self.replay_size = 32
        self.observation_shape =  (self.input_depth * self.past_range,)+ self.screen
        self.action_space_n=action_space.n

        _, self.policy, self.load_net, _ = build_network(self.observation_shape, action_space.n)

        self.load_net.compile(optimizer=Adam(lr=0.0001), loss='mse')  # clipnorm=1.
        _, _, self.load_net_guided, _ = build_guided_model(self.observation_shape, action_space.n)

        self.load_net_guided.compile(optimizer=Adam(lr=0.0001), loss='mse')  # clipnorm=1.


        self.action_space = action_space
        self.observations = np.zeros((self.input_depth * self.past_range,) + screen)

    def init_episode(self, observation):
        for _ in range(self.past_range):
            self.save_observation(observation)
        return self.observations

    def choose_action(self, observation):
        self.save_observation(observation)
        last_observations=self.observations
        #last_observations=last_observations.reshape((84,84,10))
        policy = self.policy.predict(last_observations[None, ...])[0]
        policy /= np.sum(policy)  # numpy, why?
        return np.random.choice(np.arange(self.action_space.n), p=policy)

    def save_observation(self, observation):
        
        self.observations = np.roll(self.observations, -self.input_depth, axis=0)
        self.observations[-self.input_depth:, ...] = self.transform_screen(observation)

    def transform_screen(self, data):
        return rgb2gray(imresize(data, self.screen))[None, ...]


parser = argparse.ArgumentParser(description='Evaluation of model')
parser.add_argument('--game', default='Breakout-v0', help='Name of openai gym environment', dest='game')
parser.add_argument('--evaldir', default=None, help='Directory to save evaluation', dest='evaldir')
parser.add_argument('--load_network_path',  default='', help='the path to the trained mode file')
parser.add_argument('--visualize', default=False, action='store_true')
parser.add_argument('--gbp', default=False, action='store_true', help='visualize what the network learned with Guided backpropagation')
parser.add_argument('--GradCam', action='store_false', help='visualize what the network learned with GradCam')
parser.add_argument('--duel_visual',default=False, action='store_true', help='for visualisation of dueling networks')


def main():
    args = parser.parse_args()
    # -----
    env = gym.make(args.game)
    if args.evaldir:
        env.monitor.start(args.evaldir)
    # -----
    agent = ActingAgent(env.action_space)
    model_file = args.load_network_path
    agent.load_net.load_weights(model_file)
    agent.load_net_guided.load_weights(model_file)
    if args.visualize:
        print(">> visualisation mode.")
        play_game(args, agent, env, total_episodes=1)
    else:
        game = 1
        for _ in range(10):
            done = False
            episode_reward = 0
            noops = 0

            # init game
            observation = env.reset()
            agent.init_episode(observation)
            # play one game
            print('Game #%8d; ' % (game,), end='')
            while not done:
                env.render()
                action = agent.choose_action(observation)
                observation, reward, done, _ = env.step(action)
                episode_reward += reward
                # ----
                if action == 0:
                    noops += 1
                else:
                    noops = 0
                if noops > 100:
                    break
            print('Reward %4d; ' % (episode_reward,))
            game += 1
        # -----
        if args.evaldir:
            env.monitor.close()


if __name__ == "__main__":
    main()

