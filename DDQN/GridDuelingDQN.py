from __future__ import division

import gym
import numpy as np
import random
import matplotlib.pyplot as plt

#https://github.com/awjuliani/DeepRL-Agents/blob/master/gridworld.py
from gridworld import gameEnv

from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers.convolutional import Conv2D
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.merge import _Merge, Multiply

env_size = 5
env = gameEnv(partial=False, size=env_size)

class Experience():

    def __init__(self, buffer_size):
        
        self.replay_buffer = []
        self.buffer_size = buffer_size

    def storeExperience(self, exp):

        if(len(exp)+self.buffer_size >= len(self.replay_buffer)):
            del self.replay_buffer[:(len(exp)+len(self.replay_buffer) - self.buffer_size)]

        self.replay_buffer.extend(exp)

        return self.replay_buffer

    def sample(self, sample_size):
        #return np.reshape(np.array(random.sample(self.replay_buffer, sample_size)), [sample_size, env_size])
        return random.sample(self.replay_buffer, sample_size)

class QLayer(_Merge):
    '''Q Layer that merges an advantage and value layer'''
    def _merge_function(self, inputs):
        '''Assume that the inputs come in as [value, advantage]'''
        output = inputs[0] + (inputs[1] - K.mean(inputs[1], axis=1, keepdims=True))
        return output

class QNetwork():

    def __init__(self, h_size):
        self.inputs = Input(shape=(84,84,3))
        self.actions = Input(shape=(1,), dtype='int32')
        self.actions_onehot = Lambda(K.one_hot, arguments={'num_classes':env.actions}, output_shape=(None, env.actions))(self.actions)

        x = Conv2D(filters=32, kernel_size=[8,8], strides=[4,4], input_shape=(84, 84, 3))(self.inputs)
        x = Conv2D(filters=64, kernel_size=[4,4],strides=[2,2])(x)
        x = Conv2D(filters=64, kernel_size=[3,3],strides=[1,1])(x)
        x = Conv2D(filters=h_size, kernel_size=[7,7],strides=[1,1])(x)

        #Splice outputs of last conv layer using lambda layer
        x_value = Lambda(lambda x: x[:,:,:,:h_size//2])(x)
        x_advantage = Lambda(lambda x: x[:,:,:,h_size//2:])(x)

        #Process spliced data stream into value and advantage function
        value = Dense(env.actions, activation="linear")(x_value)
        advantage = Dense(env.actions, activation="linear")(x_advantage)

        #Recombine value and advantage layers into Q layer
        q = QLayer()([value, advantage])

        self.q_out = Multiply()([q, self.actions_onehot])
        self.q_out = Lambda(lambda x: K.cumsum(x, axis=3), output_shape=(1,))(self.q_out)
        #need to figure out how to represent actions within training
        self.model = Model(inputs=[self.inputs, self.actions], outputs=[q, self.q_out])
        self.model.compile(optimizer="Adam", loss="mean_squared_error")

        self.model.summary()



def resizeFrames(states):
    return np.reshape(states, [84*84*3])


h_size = 512
batch_size = 32
update_freq = 4
gamma = 0.9
start_eps = 1.
end_eps = 0.1
annealing_steps = 10000.
num_episodes = 10000
pre_train_steps = 10000
max_episode_length = 50
target_update_rate = 0.001

eps = start_eps
step_drop = (start_eps - end_eps) / annealing_steps

#store rewards and steps per episode
j_list = []
r_list = []
total_steps = 0

actor_network = QNetwork(h_size)
target_network = QNetwork(h_size)

experience = Experience(buffer_size=50000)

## Do this to periodically update the target network with ##
## the weights of the actor network                                   ##
#target_network.set_weights(actor_network.get_weights())


for i in xrange(num_episodes):
    episode_exp = Experience(buffer_size=50000)
    s = env.reset()
    #s = resizeFrames(s)
    done = False
    total_reward = 0
    j = 0

    while j < max_episode_length:
        j += 1

        if np.random.rand(1) < eps or total_steps < pre_train_steps:
            a = np.random.randint(0, 4)
        else:
            prediction = actor_network.model.predict([s.reshape((1,84,84,3)), np.zeros((32, 1))])
            a = np.argmax(prediction[0])

        s1, r, done = env.step(a)
        #s1 = resizeFrames(s1)
        total_steps += 1
        episode_exp.storeExperience(np.reshape(np.array([s,a,r,s1,done]), [1, 5]))

        if total_steps > pre_train_steps:
            if eps > end_eps:
                eps -= step_drop

            if total_steps % update_freq == 0:
                train_batch = experience.sample(batch_size)
                
                #Have to do this because couldn't easily splice array
                #out of experience buffer, e.g.,
                #train_input = train_batch[:, 3]
                #when train_batch was a numpy array
                this_state = np.ndarray((batch_size, 84, 84, 3))
                actions = np.ndarray((batch_size, 1))
                rewards = np.ndarray((batch_size, 1))
                next_state = np.ndarray((batch_size, 84, 84, 3))
                dones = np.ndarray((batch_size, 1))
                for i in range(batch_size):
                    this_state[i] = train_batch[i][0]
                    actions[i] = train_batch[i][1]
                    rewards[i] = train_batch[i][2]
                    next_state[i] = train_batch[i][3]
                    dones[i] = train_batch[i][4]
            
                q1 = actor_network.model.predict([next_state, np.zeros((32, 1))])
                q1 = np.argmax(q1[0], axis=3)

                q2 = target_network.model.predict([next_state, np.zeros((32, 1))])
                q2 = q2[0].reshape((batch_size, env.actions))

                end_multiplier = -(dones - 1)

                double_q = q2[range(32), q1.reshape((32))].reshape((32, 1))

                target_q = rewards + (gamma*double_q*end_multiplier)

                print ("Target Q Shape: ", target_q.shape)
                q = actor_network.model.predict([this_state, actions])
                #q_of_actions = q[:, train_batch[:, 1]]
                print (target_q.shape)
                actor_network.model.fit([this_state, actions], [np.zeros((32, 1, 1, 4)) ,target_q])
                target_network.model.set_weights(actor_network.model.get_weights())


        total_reward += r
        s = s1

        if done == True:
            break

    experience.storeExperience(episode_exp.replay_buffer)
    j_list.append(j)
    r_list.append(total_reward)