from __future__ import print_function
import warnings ; warnings.filterwarnings('ignore') # mute warnings, live dangerously

import matplotlib.pyplot as plt
import matplotlib as mpl ; mpl.use("Agg")
import matplotlib.animation as manimation

import numpy as np

import gym, os, sys, time, argparse
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense
from keras import backend as K
from keras.optimizers import Adam

sys.path.append('..')

from skimage.transform import resize # preserves single-pixel info _unlike_ img = img[::2,::2]
from skimage.color import rgb2gray
import backpropagation
import grad_cam
import time

FRAME_WIDTH = 84  # Resized frame width
FRAME_HEIGHT = 84  # Resized frame height
STATE_LENGTH = 4 # How many Frames di we will stack for one state


preprocess = lambda img: np.uint8(resize(rgb2gray(img), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)

def build_network(num_actions):
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT))) #subsample=strides
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_actions))

    model.summary()
    return model#, q_values



def rollout(model, env, max_ep_len=3e3, render=False):

    history = {'ins': [], 'state': [], 'action': [], 'outs': [], 'hx': [], 'cx': []}

    state = preprocess(env.reset()) # get first state
    state = [state for _ in range(STATE_LENGTH)]
    state = np.stack(state, axis=0)
    state2 = np.expand_dims(np.asarray(state).astype(np.float64), axis=0)

    episode_length, epr, eploss, done  = 0, 0, 0, False # bookkeeping
    state3=state2
    while not done and episode_length <= max_ep_len:
        episode_length += 1
        print(np.argmax(model.predict(np.float32(state2 / 255.0))))
        action=np.argmax(model.predict(np.float32(state2 / 255.0)))
        if(np.array_equal(state3, state2)):
            print("Oh Oh !\n\n\n")
        state3=state2
        obs, reward, done, expert_policy = env.step(action)
        env.render()
        state_new = preprocess(obs) ; epr += reward
        state_new = np.expand_dims(np.asarray(state_new).astype(np.float64), axis=0)
        state = np.append(state[1:, :, :], state_new, axis=0)
        
        
        state2 = np.expand_dims(np.asarray(state).astype(np.float64), axis=0)

         #save info!
        history['ins'].append(obs)
        history['state'].append(state2)
        history['action'].append(action)
        print('\tstep # {}, reward {:.0f}'.format(episode_length, epr), end='\r')

    return history


def preprocessImage():
    state = preprocess(obs)
    state = [state for _ in range(STATE_LENGTH)]
    state = np.stack(state, axis=0)
    state = np.expand_dims(np.asarray(state).astype(np.float64), axis=0)

    return True





def make_movie(env_name='Breakout-v0',  num_frames=200, first_frame=0, resolution=75, \
                save_dir='./movies/', prefix='default', density=5, radius=5,  overfit_mode=False):
    

    env = gym.make(env_name)
    action_num=env.action_space.n
    model=build_network(action_num)#load_dir, checkpoint=checkpoint)
    model.load_weights("dqn_model_weights.hdf5")
    guided_model = backpropagation.build_guided_model(action_num)
    guided_model.load_weights("dqn_model_weights.hdf5")
    # get a rollout of the policy
    movie_title ="{}-{}-{}.mp4".format(prefix, num_frames, env_name.lower())
    max_ep_len = first_frame + num_frames + 1
    history = rollout(model, env, max_ep_len=max_ep_len)
    env.close()
    # make the movie!
    start = time.time()
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title=movie_title, artist='mateus', comment='atari-video')
    writer = FFMpegWriter(fps=8, metadata=metadata)

    prog = '' ; total_frames = len(history['state'])
    dim = np.zeros((FRAME_WIDTH,FRAME_HEIGHT))
    f = plt.figure(figsize=[6, 6*1.3], dpi=resolution)
    with writer.saving(f, save_dir + movie_title, resolution):
        for i in range(num_frames):
            ix = first_frame+i
            print(ix)
            if ix < total_frames: # prevent loop from trying to process a frame ix greater than rollout length
                frame = history['state'][ix].copy()
                action = history['action'][ix].copy()
                gbp_heatmap = backpropagation.guided_backprop(guided_model,"dense_4", frame)
                geb_gradCam = grad_cam.grad_cam(model,"dense_2",frame, action) 
                gbp_heatmap_pic=gbp_heatmap[0][0]
                #gbp_heatmap_pic-= gbp_heatmap_pic.mean()
                #gbp_heatmap_pic/= (gbp_heatmap_pic.std() + 1e-5)
                #gbp_heatmap_pic*= 0.1
                frame=frame[0][0]

    # clip to [0, 1]
                #gbp_heatmap_pic += 0.5
                #print(gbp_heatmap_pic)
                gbp_heatmap_pic = np.clip(gbp_heatmap_pic, 0, 1)
                frame=frame/255
                frame=np.clip(frame,0,1)
                #print(frame)
                #dim = np.zeros((FRAME_WIDTH,FRAME_HEIGHT))
                mixed = np.stack((gbp_heatmap_pic,gbp_heatmap_pic, frame), axis=2)
                #print(gbp_heatmap_pic)
                #print("GBP Heatmap shape: {}".format(gbp_heatmap_pic.shape))
                #plt.imshow(backpropagation.deprocess_image(gbp_heatmap_pic))
                #plt.show()
                ##frame= np.array_split(frame, [1], axis=1)  
                plt.imshow(mixed) ; #plt.title(env_name.lower(), fontsize=15)
                #print(frame.shape, "ttttttt\n")
                writer.grab_frame() ; f.clear()
                
                #tstr = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start))
                #print('\ttime: {} | progress: {:.1f}%'.format(tstr, 100*i/min(num_frames, total_frames)), end='\r')
    print('\nfinished.')
    


if __name__ == '__main__':

    np.set_printoptions(threshold=sys.maxsize)

    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-e', '--env', default='Breakout-v4', type=str, help='gym environment')
    parser.add_argument('-f', '--num_frames', default=12, type=int, help='number of frames in movie')
    parser.add_argument('-i', '--first_frame', default=0, type=int, help='index of first frame')
    parser.add_argument('-dpi', '--resolution', default=75, type=int, help='resolution (dpi)')
    parser.add_argument('-s', '--save_dir', default='./movies/', type=str, help='dir to save agent logs and checkpoints')
    parser.add_argument('-p', '--prefix', default='default', type=str, help='prefix to help make video name unique')
    
    args = parser.parse_args()

    make_movie(args.env, args.num_frames, args.first_frame,   args.resolution,args.save_dir, args.prefix)
