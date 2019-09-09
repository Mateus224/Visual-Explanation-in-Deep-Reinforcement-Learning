# coding:utf-8
import sys
import os
import gym
import random
import keras
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.layers import Convolution2D, Flatten, Dense
from keras.models import load_model, Sequential, Model
#from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers import merge, Input
from keras import backend as K
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger


ENV_NAME = 'BreakoutDeterministic-v4'  # Environment name
FRAME_WIDTH = 84  # Resized frame width
FRAME_HEIGHT = 84  # Resized frame height
NUM_EPISODES = 40000  # Number of episodes the agent plays
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
GAMMA = 0.99  # Discount factor
EXPLORATION_STEPS =800000   # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
INITIAL_REPLAY_SIZE = 50000  # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 500000  # Number of replay memory the agent uses for training
BATCH_SIZE = 32  # Mini batch size
TARGET_UPDATE_INTERVAL = 10000  # The frequency with which the target network is updated
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
SAVE_INTERVAL = 100000  # The frequency with which the network is saved
NO_OP_STEPS = 25  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
LOAD_NETWORK = False
TRAIN = True
SAVE_NETWORK_PATH = 'saved_networks/' + ENV_NAME
SAVE_SUMMARY_PATH = 'summary/' + ENV_NAME
NUM_EPISODES_AT_TEST = 80  # Number of episodes the agent plays at test time

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class Agent():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.t = 0
        self.history = LossHistory()

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        # Create replay memory
        self.replay_memory = deque()

        # Create q network
        self.q_network = self.build_network()
        q_network_weights = self.q_network.trainable_weights

        # Create target network
        self.target_network = self.build_network()
        target_network_weights = self.target_network.trainable_weights

        # Define target network update operation
        #self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        # Define loss and gradient update operation
        #self.a, self.y, self.loss, self.grads_update = self.build_training_op(q_network_weights)


        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        #self.summary_writer = tf.summary.FileWriter(SAVE_SUMMARY_PATH, self.sess.graph)

        #if not os.path.exists(SAVE_NETWORK_PATH):
        #    os.makedirs(SAVE_NETWORK_PATH)

        #self.sess.run(tf.initialize_all_variables())

        # Load network
        if LOAD_NETWORK:
            self.load_network()

        # Initialize target network
        #self.sess.run(self.update_target_network)

    def update_weights(self):
        self.target_network.set_weights(self.q_network.get_weights())

        

    def build_network(self):
        input_layer = Input(shape = ( 4 ,84, 84), name='input')
        conv1 = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', name='conv1')(input_layer)
        conv2 = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu', name='conv2')(conv1)
        conv3 = Convolution2D(64, 3, 3, activation = 'relu', name='conv3')(conv2)
        flatten = Flatten(name='flatten1')(conv3)
        fc1 = Dense(512, name='dense1')(flatten)
        advantage = Dense(self.num_actions, name='denseAdvan')(fc1)
        fc2 = Dense(512, name='densefc2')(flatten)
        value = Dense(1, name='denseValue' )(fc2)
        #policy = merge([advantage, value], mode = lambda x: x[0]-K.mean(x[0])+x[1], output_shape = (self.n_actions,))
        self.q_values = merge([advantage, value], name='merge', mode = lambda x: x[0]-K.mean(x[0])+x[1], output_shape = (self.num_actions,))
        #best_action = tf.argmax(self.q_values, 1)


         #select_q_value_of_action = merge([q_value_prediction,action_one_hot], mode="mul")
        #target_q_value = Lambda(lambda x:K.sum(x, axis=-1, keepdims=True),output_shape=lambda_out_shape)(select_q_value_of_action)

        model = Model(input=[input_layer], output=[self.q_values])
        model.compile(loss='mse', optimizer=Adam(lr=0.00001)) #0.0000625 #0.000001#loss=huber_loss, loss='mse'


        print("Successfully constructed networks.")

        model.summary()


        

        return model

    #def huber_loss(a, b, in_keras=True):
    #    error = a - b
    #    quadratic_term = error*error / 2
    #    linear_term = abs(error) - 1/2
    #    use_linear_term = (abs(error) > 1.0)
    #if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
    #    use_linear_term = K.cast(use_linear_term, 'float32')
    #return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term


    def get_initial_state(self, observation, last_observation):
        #processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(rgb2gray(observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
        
        state = [processed_observation for _ in range(STATE_LENGTH)]
        
        return np.stack(state, axis=0)

    def get_action(self, state):
        if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
            action = random.randrange(self.num_actions)
        else:
            #action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
            state=np.expand_dims(state, axis=0)
            action = np.argmax(self.q_network.predict(np.float32(state/255.0)))
        # Anneal epsilon linearly over time
        if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        return action

    def run(self, state, action, reward, terminal, next_state):
        #print(observation.shape, state.shape)
        next_state = np.append(state[1:, :, :], next_state, axis=0)
        

        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        reward = np.clip(reward, -1, 1)
        #print(reward,"\n")

        # Store transition in replay memory
        self.replay_memory.append((state, action, reward, next_state, terminal))
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()

        if self.t >= INITIAL_REPLAY_SIZE:
            # Train network
            if self.t % TRAIN_INTERVAL == 0:
                self.train_network()

            # Update target network
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.update_weights()

            # Save network
            if self.t % SAVE_INTERVAL == 0:
                self.q_network.save("DDQN_model_weights.hdf5")
                #print(self.history.loss)
                # Plot training & validation loss values
                #plt.plot(self.history.history['loss'])
                #plt.title('Model loss')
                #plt.ylabel('Loss')
                #plt.xlabel('Epoch')
                #plt.legend(['Train', 'Test'], loc='upper left')
                #plt.show()

                #save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=self.t)
                print('Successfully saved: ')# + save_path)

        self.total_reward += reward
        #self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
        state=np.expand_dims(state, axis=0)
        self.total_q_max +=np.max(self.q_network.predict(np.float32(state/255.0)))
        self.duration += 1

        if terminal:
            # Write summary
            #if self.t >= INITIAL_REPLAY_SIZE:
                #stats = [self.total_reward, self.total_q_max / float(self.duration),
                #        self.duration, self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL))]
                #for i in range(len(stats)):
                    #self.sess.run(self.update_ops[i], feed_dict={
                    #    self.summary_placeholders[i]: float(stats[i])
                    #})
                #summary_str = self.sess.run(self.summary_op)
                #self.summary_writer.add_summary(summary_str, self.episode + 1)

            # Debug
            if self.t < INITIAL_REPLAY_SIZE:
                mode = 'random'
            elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'
            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.0f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_q_max / float(self.duration),
                self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)), mode))

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1

        return next_state

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        #for state, action, reward, next_state, done in minibatch:
        target_q_values = self.target_network.predict(np.array(next_state_batch) / 255.0, batch_size=32)
        #print(target_q_values, "target_q_values\n")
        y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values, axis=1)
        target_q_values[0][np.array(action_batch)]=y_batch
        #print(target_q_values,"test\n")
        #print(np.array(state_batch).shape,"\n shape\n")
        csv_logger = CSVLogger('log.csv', append=True, separator=';')
        self.q_network.fit(np.array(state_batch), target_q_values, nb_epoch=1, verbose=0, batch_size=32, callbacks=[csv_logger])


    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Total Reward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Max Q/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Average Loss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def load_network(self):
        self.q_network.load_weights("DDQN_model_weights.hdf5")


    def get_action_at_test(self, state):
        if random.random() <= 0.05:
            state=np.expand_dims(state, axis=0)
            action = random.randrange(self.num_actions)
        else:
            #print(np.float32(state / 255.0))
            #action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))
            state=np.expand_dims(state, axis=0)
            action = np.argmax(self.q_network.predict(state/255.0))
        self.t += 1

        return action


def preprocess(observation, last_observation):
    #processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(rgb2gray(observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
    return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))



def main():
    env = gym.make(ENV_NAME)
    agent = Agent(num_actions=env.action_space.n)

    if TRAIN:  # Train mode
        for _ in range(NUM_EPISODES):
            terminal = False
            observation = env.reset()
            for _ in range(random.randint(1, NO_OP_STEPS)):
                last_observation = observation
                observation, _, _, _ = env.step(1)  # Do nothing
            state = agent.get_initial_state(observation, last_observation)
            while not terminal:
                last_state = state
                action = agent.get_action(last_state)
                observation, reward, terminal, _ = env.step(action)
                # env.render()
                state = preprocess(observation, last_observation)
                state = agent.run(last_state, action, reward, terminal, state)

    else:  # Test mode
        # env.monitor.start(ENV_NAME + '-test')
        for _ in range(NUM_EPISODES_AT_TEST):
            terminal = False
            observation = env.reset()
            for _ in range(random.randint(1, NO_OP_STEPS)):
                last_observation = observation
                observation, _, _, _ = env.step(0)  # Do nothing
            state = agent.get_initial_state(observation, last_observation)
            while not terminal:
                last_observation = observation
                action = agent.get_action_at_test(state)
                observation, _, terminal, _ = env.step(action)
                env.render()
                processed_observation = preprocess(observation, last_observation)
                state = np.append(state[1:, :, :], processed_observation, axis=0)
        # env.monitor.close()


if __name__ == '__main__':
    np.set_printoptions(threshold=sys.maxsize)
    main()
