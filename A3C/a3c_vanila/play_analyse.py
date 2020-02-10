import argparse
import numpy as np
#from environment import Environment
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Convolution2D, Flatten, Dense

from visualization.backpropagation import *
from PIL import Image
from visualization.grad_cam import *
#import visualization.grad_cam.py


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

def parse():
    parser = argparse.ArgumentParser(description="DQN")
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')
    parser.add_argument('--gbp', action='store_true', help='visualize what the network learned with Guided backpropagation')
    parser.add_argument('--gradCAM', action='store_true', help='visualize what the network learned with GradCAM')
    parser.add_argument('--gbp_GradCAM', action='store_true', help='visualize what the network learned with Guided GradCAM')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def init_saliency_map(args, agent, history, first_frame=0, num_frames=1000, prefix='QF_', resolution=75, save_dir='./movies/', env_name='Breakout-v0'):

    _, policy_model, load_model_guided ,_= build_network(agent.observation_shape, agent.action_space_n)
    _, policy_model,load_model_grad_cam,_ = build_network(agent.observation_shape, agent.action_space_n)

    load_model_guided.load_weights(args.load_network_path)
    load_model_grad_cam.load_weights(args.load_network_path)
    frame_1= np.zeros((84, 84))
    total_frames=len(history['state'])
    backprop_fn = init_guided_backprop(load_model_guided,"dense_6")
    gradient_fn = init_grad_cam(load_model_grad_cam, "convolution2d_9")
    if args.duel_visual:
        backprop_fn_advatage = init_guided_backprop(visualization_network_model,"Attention V")
        fig_array = np.zeros((2,2,num_frames,84,84,3))
    else:
        fig_array = np.zeros((1,2,num_frames,84,84,3))
    for i in range(num_frames):#total_frames): #num_frames
        ix = first_frame+i
        if ix < total_frames: # prevent loop from trying to process a frame ix greater than rollout length
            frame = history['state'][ix].copy()
            action = history['action'][ix]#.copy()
            frame = np.expand_dims(frame, axis=0)
            if ix%10==0:
                print(ix)
            #gbp_heatmap = guided_backprop(frame, backprop_fn)
            gradCam_heatmap = grad_cam(gradient_fn, frame, action)
            gradCam_heatmap = np.asarray(gradCam_heatmap)
            history['cam'].append(gradCam_heatmap)
            if args.duel_visual:
                gbp_heatmap = guided_backprop(frame, backprop_fn_advatage)
                history['gradients_duel_adv'].append(gbp_heatmap)
    history_grad = history['gradients'].copy()
    history_cam = history['cam'].copy()
    fig_array[0,0],fig_array[0,1] = normalization(history_cam, history, "cam")
    if args.duel_visual:
        history_grad_adv=history['gradients_duel_adv'].copy()
        fig_array[1,0],fig_array[1,1] = normalization(history_grad_adv, history, "gdb")
    make_movie(args,history,fig_array,first_frame,num_frames,resolution,save_dir,prefix,env_name)

#def normalization_cam(cam_heatmap,history)


def normalization(heatmap, history, visu):
    heatmap=np.asarray(heatmap)
    if visu=='gdb':
        heatmap = heatmap[:,0,:,:,:]
        #gbp_heatmap_pic=gbp_heatmap[0,:,:,:]
        heatmap-= heatmap.mean() 
        heatmap/= (heatmap.std() + 1e-5) #
        heatmap*= 0.1 


        # clip to [0, 1]
        #gbp_heatmap += 0.5
        heatmap = np.clip(heatmap, -1, 1)
        heatmap_pic1 = heatmap[:,0,:,:,9]
        heatmap_pic2 = heatmap[:,0,:,:,0]
    if visu=='cam':
        #print(heatmap.shape)
        #heatmap = heatmap[:,0,:,:,:]
        #heatmap-= heatmap.mean() 
        #heatmap/= (heatmap.std() + 1e-5) #
        #heatmap*= 0.1 
        heatmap = np.clip(heatmap, 0, 1)

        
        heatmap_pic1 = heatmap[:,:,:]

        heatmap_pic2 = heatmap[:,:,:]
    all_unproc_frames = history['un_proc_state'].copy()
    frame=np.zeros((num_frames,84,84,3))
    for i in range(len(all_unproc_frames)):
        frame[i,:,:,:]=np.asarray(Image.fromarray(all_unproc_frames[i]).resize((84, 84), Image.BILINEAR))/255
    proc_frame1 = overlap(frame,heatmap_pic1)
    proc_frame2 = overlap(frame,heatmap_pic2)

    

    #print(np.asarray(frame))
    #frame = frame[:,:,:,0]
    #mixed = np.stack((gbp_heatmap_pic1, frame, gbp_heatmap_pic1), axis=3) 
    return proc_frame1, proc_frame2


def overlap(frame,gbp_heatmap):
    color_neg = [1.0, 0.0, 0.0]
    color_pos = [0.0, 1.0, 0.0]
    color_chan = np.ones((num_frames,84,84,2),dtype=gbp_heatmap.dtype)
    alpha = 0.5
    #beta = 0.25
    #gbp_heatmap = np.expand_dims(gbp_heatmap, axis=3)
    _gbp_heatmap = [gbp_heatmap for _ in range(3)]
    _gbp_heatmap=np.stack(_gbp_heatmap,axis=3)
    gbp_heatmap=_gbp_heatmap
    #gbp_heatmap = np.concatenate((gbp_heatmap,color_chan),axis=3)
    gbp_heatmap_pos=np.asarray(gbp_heatmap.copy())
    gbp_heatmap_neg=np.asarray(gbp_heatmap.copy())
    gbp_heatmap_pos[gbp_heatmap_pos<0.0]=0
    gbp_heatmap_neg[gbp_heatmap_neg>=0.0]=0
    gbp_heatmap_neg=-gbp_heatmap_neg
    gbp_heatmap = color_pos * gbp_heatmap_pos[:,:,:,:] + color_neg * gbp_heatmap_neg[:,:,:,:]
    #gbp_heatmap = color_pos * gbp_heatmap_pos[:,:,:,:] + color_neg * gbp_heatmap_neg[:,:,:,:]
    mixed = alpha * gbp_heatmap + (1.0 - alpha) * frame
    mixed = np.clip(mixed,0,1)



    return mixed
    #return mixed

def make_movie(args,history,fig_array,first_frame,num_frames,resolution,save_dir,prefix,env_name):
    movie_title ="{}-{}-{}.mp4".format(prefix, num_frames, env_name.lower())
    max_ep_len = first_frame + num_frames + 1
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='test', artist='mateus', comment='atari-video')
    writer = FFMpegWriter(fps=8, metadata=metadata)
    total_frames = len(history['state'])
    fig = plt.figure(figsize=[6, 6*1.3], dpi=resolution)
    with writer.saving(fig, save_dir + movie_title, resolution):
        for i in range(121):#total_frames): #num_frames
            plotColumns = 2
            plotRows = 1
            if args.duel_visual:
                titleList=["V(s; theta, beta)","A(s,a;thata,alpha)"]
                for j in range(0, plotColumns*plotRows):
                    img = fig_array[j,i,:,:,:]
                    ax=fig.add_subplot(plotRows, plotColumns, j+1)
                    ax.set_xlabel(titleList[j])
                    plt.imshow(img)
            else:
                plt.imshow(fig_array[0,i,:,:,:]) #if error because no directory that exist
            writer.grab_frame() 
            fig.clear()
            if i%100==0:
                print(i)



def play_game(args, agent, env, total_episodes=1):
    
    history = { 'state': [], 'action': [], 'gradients':[], 'gradients_duel_adv':[],'movie_frames':[]}
    rewards = []
    agent.load_net.load_weights(args.load_network_path)
    for i in range(total_episodes):
        state = env.reset()
        #agent.init_game_setting()
        done = False
        episode_reward = 0.0
        action_state=agent.init_episode(state)
        #playing one game
        #while(not done):
        for _ in range(121):
            history['state'].append(action_state)
            action = agent.choose_action(state)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            agent.save_observation(state)
            action_state=agent.observations
            history['action'].append(action)
        rewards.append(episode_reward)
    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))
    init_saliency_map(args, agent, history)

    return history


def test(agent, env, total_episodes=30):
    rewards = []
    for i in range(total_episodes):
        state = env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        #playing one game
        while(not done):
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))


def run(args):
    if args.test_dqn:
        env = Environment('SeaquestNoFrameskip-v0', args, atari_wrapper=True, test=True)
        from agent_dir.agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        play_game(agent, env, total_episodes=1)






if __name__ == '__main__':
    args = parse()
    run(args)
