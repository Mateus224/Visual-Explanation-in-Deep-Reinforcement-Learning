import argparse
import numpy as np
#from environment import Environment
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
import traceback

from visualization.backpropagation import *
from PIL import Image
from visualization.grad_cam import *
from visualization.model import build_network
#import visualization.grad_cam.py

num_frames=60


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


def init_saliency_map(args, agent, history, first_frame=0, prefix='QF_', resolution=75, save_dir='./movies/', env_name='Breakout-v0'):

    #_, _, load_model ,_= build_network(agent.observation_shape, agent.action_space_n)
    #_, _,load_guided_model,_ = build_guided_model(agent.observation_shape, agent.action_space_n)
    print(args.load_network_path)

    #load_model.load_weights(args.load_network_path)
    #load_guided_model.load_weights(args.load_network_path)


    total_frames=len(history['state'])
    backprop_actor = init_guided_backprop(agent.load_net,"timedistributed_5")
    backprop_critic = init_guided_backprop(agent.load_net,"timedistributed_5")
    cam_actor = init_grad_cam(agent.load_net, "timedistributed_3")
    cam_critic = init_grad_cam(agent.load_net, "timedistributed_3", False)
    guidedBackprop_actor = init_guided_backprop(agent.load_net_guided,"timedistributed_12")
    guidedBackprop_critic = init_guided_backprop(agent.load_net_guided,"timedistributed_12")
    gradCAM_actor = init_grad_cam(agent.load_net_guided, "timedistributed_10")
    gradCAM_critic = init_grad_cam(agent.load_net_guided, "timedistributed_10", False)
    fig_array = np.zeros((4,2,num_frames,84,84,3))
    for i in range(num_frames):#total_frames): #num_frames
        ix = first_frame+i
        if ix < total_frames: # prevent loop from trying to process a frame ix greater than rollout length
            frame = history['state'][ix].copy()
            action = history['action'][ix]#.copy()
            frame = np.expand_dims(frame, axis=0)
            if ix%10==0:
                print(ix)

            actor_gbp_heatmap = guided_backprop(frame, backprop_actor)
            actor_gbp_heatmap = np.asarray(actor_gbp_heatmap)
            history['gradients_actor'].append(actor_gbp_heatmap)

            #print(np.array(frame))

            actor_gbp_heatmap = guided_backprop(frame, backprop_critic)
            actor_gbp_heatmap = np.asarray(actor_gbp_heatmap)
            history['gradients_critic'].append(actor_gbp_heatmap)

            Cam_heatmap = grad_cam(cam_actor, frame, action)
            Cam_heatmap = np.asarray(Cam_heatmap)
            history['gradCam_actor'].append(Cam_heatmap)

            gradCam_heatmap = grad_cam(cam_critic, frame, action, False)
            gradCam_heatmap = np.asarray(gradCam_heatmap)
            history['gradCam_critic'].append(gradCam_heatmap)

            actor_gbp_heatmap = guided_backprop(frame, guidedBackprop_actor)
            actor_gbp_heatmap = np.asarray(actor_gbp_heatmap)
            history['gdb_actor'].append(actor_gbp_heatmap)
            
            critic_gbp_heatmap = guided_backprop(frame, guidedBackprop_critic)
            critic_gbp_heatmap = np.asarray(critic_gbp_heatmap)
            history['gdb_critic'].append(critic_gbp_heatmap)

            gradCam_heatmap = grad_cam(gradCAM_actor, frame, action)
            gradCam_heatmap = np.asarray(Cam_heatmap)
            history['guidedGradCam_actor'].append(Cam_heatmap)

            gradCam_heatmap = grad_cam(gradCAM_critic, frame, action, False)
            gradCam_heatmap = np.asarray(gradCam_heatmap)
            history['guidedGradCam_critic'].append(gradCam_heatmap)

            


    history_gradients_actor = history['gradients_actor'].copy()
    history_gradients_critic = history['gradients_critic'].copy()
    history_gradCam_actor = history['gradCam_actor'].copy()
    history_gradCam_critic = history['gradCam_critic'].copy()
    history_gdb_actor = history['gdb_actor'].copy()
    history_gdb_critic = history['gdb_critic'].copy()
    history_guidedGradCam_actor = history['guidedGradCam_actor'].copy()
    history_guidedGradCam_critic = history['guidedGradCam_critic'].copy()
    fig_array[0,0] = normalization(history_gradients_actor, history, "gdb",GDB_actor=0)
    fig_array[0,1] = normalization(history_gradients_critic, history, 'gdb')
    fig_array[1,0] = normalization(history_gradCam_actor, history, "cam", )
    fig_array[1,1] = normalization(history_gradCam_critic, history, 'cam')
    fig_array[2,0] = normalization(history_gdb_actor, history, "gdb", GDB_actor=0)
    fig_array[2,1] = normalization(history_gdb_critic, history, 'gdb')
    fig_array[3,0] = normalization(history_guidedGradCam_actor, history, "cam")
    fig_array[3,1] = normalization(history_guidedGradCam_critic, history, 'cam')

    make_movie(args,history,fig_array,first_frame,num_frames,resolution,save_dir,prefix,env_name)

#def normalization_cam(cam_heatmap,history)


#def save_for_GradCam(heatmap, gdp=1)



def normalization(heatmap, history, visu, GDB_actor=0):
    heatmap=np.asarray(heatmap)
    if visu=='gdb':
        print(heatmap.shape)
        heatmap = heatmap[:,:,:]
        #gbp_heatmap_pic=gbp_heatmap[0,:,:,:]
        heatmap-= heatmap.mean() 
        heatmap/= (heatmap.std() + 1e-5) #
        if (GDB_actor):
            #print(heatmap)
            heatmap*=50
        else:
            heatmap*= 0.1 #0.1 


        # clip to [0, 1]
        #gbp_heatmap += 0.5
        heatmap = np.clip(heatmap, -1, 1)
        print("gdb",heatmap.shape)
        heatmap_pic1 = heatmap[:,0,9,:,:]
        #save_for_GradCam(heatmap_pic1, 1)
    if visu=='cam':
        
        #print(heatmap.shape)
        #heatmap = heatmap[:,0,:,:,:]
        #heatmap-= heatmap.mean() 
        #heatmap/= (heatmap.std() + 1e-5) #
        heatmap*= 1
        heatmap = np.clip(heatmap, 0, 1)
        print("can",heatmap.shape)
        
        heatmap_pic1 = heatmap[:,:,:]
        print("heatmapCAM",heatmap_pic1.shape)
        #save_for_GradCam(heatmap_pic1)


    all_unproc_frames = history['un_proc_state'].copy()
    frame=np.zeros((num_frames,84,84,3))
    for i in range(len(all_unproc_frames)):
        frame[i,:,:,:]=np.asarray(Image.fromarray(all_unproc_frames[i]).resize((84, 84), Image.BILINEAR))/255
    proc_frame1 = overlap(frame,heatmap_pic1)
    

    #print(np.asarray(frame))
    #frame = frame[:,:,:,0]
    #mixed = np.stack((gbp_heatmap_pic1, frame, gbp_heatmap_pic1), axis=3) 
    return proc_frame1


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
        titleListX=["Actor","Critic"]
        titleListY=["Backpropagation", "GradCam", "Guided Backpropagation","Guided GeadCam"]
        for i in range(num_frames):#total_frames): #num_frames
            plotColumns = np.shape(fig_array)[1] #fig_array[0,:,0,0,0,0].shape
            plotRows = np.shape(fig_array)[0]#fig_array[:,0,0,0,0,0].shape
            z=0
            
            for j in range(0, plotRows):
                for k in range(0, plotColumns):
                    img = fig_array[j,k,i,:,:,:]
                    ax=fig.add_subplot(plotRows, plotColumns, z+1)
                    ax.set_ylabel(titleListY[j])
                    ax.set_xlabel(titleListX[k])
                    plt.imshow(img)
                    z=z+1

            writer.grab_frame() 
            fig.clear()
            if i%100==0:
                print(i)



def play_game(args, agent, env, total_episodes=1):
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    history = { 'state': [], 'un_proc_state' : [], 'action': [], 'gradients_actor':[], 'gradients_critic':[],'gradCam_actor':[],'gradCam_critic':[], 'gdb_actor':[],'gdb_critic':[], 'guidedGradCam_actor':[],'guidedGradCam_critic':[] ,'movie_frames':[]}
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
        for _ in range(num_frames):
            history['state'].append(action_state)
            history['un_proc_state'].append(state)
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
