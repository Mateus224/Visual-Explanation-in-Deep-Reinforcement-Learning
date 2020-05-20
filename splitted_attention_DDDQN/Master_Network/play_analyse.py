import argparse
import numpy as np
#from environment import Environment
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
from deeprl_prj.dqn_keras import *

from visualization.backpropagation import *
from visualization.model import build_network
from PIL import Image
from visualization.grad_cam import *

num_frames=600

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


def init_saliency_map(args, agent, history, first_frame=0, num_frames=num_frames, prefix='QF_', resolution=450, save_dir='./movies/', env_name='Breakout-v0'):
    #load_model = create_model(agent.input_shape, agent.num_actions, agent.net_mode, args, "QNet")
    load_model = build_network(agent.input_shape, agent.num_actions)
    load_guided_model = build_guided_model(agent.input_shape, agent.num_actions)

    load_model.load_weights(args.load_network_path)
    load_guided_model.load_weights(args.load_network_path)
    total_frames=len(history['state'])
    backprop_actor = init_guided_backprop(load_model,"Attention A")
    backprop_critic = init_guided_backprop(load_model,"Attention V")
    cam_actor = init_grad_cam(load_model, "timedistributed_16")
    cam_critic = init_grad_cam(load_model, "timedistributed_16", False)
    guidedBackprop_actor = init_guided_backprop(load_guided_model,"Attention A")
    guidedBackprop_critic = init_guided_backprop(load_guided_model,"Attention V")
    gradCAM_actor = init_grad_cam(load_guided_model, "timedistributed_23")
    gradCAM_critic = init_grad_cam(load_guided_model, "timedistributed_23", False)
    fig_array = np.zeros((6,2,num_frames,84,84,3))
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
            gradCam_heatmap = np.asarray(gradCam_heatmap)
            history['guidedGradCam_actor'].append(gradCam_heatmap)

            gradCam_heatmap = grad_cam(gradCAM_critic, frame, action, False)
            gradCam_heatmap = np.asarray(gradCam_heatmap)
            history['guidedGradCam_critic'].append(gradCam_heatmap)

            


    history_gradients_actor = history['gradients_actor'].copy()
    history_gradients_critic = history['gradients_critic'].copy()
    history_gdb_actor = history['gdb_actor'].copy()
    history_gdb_critic = history['gdb_critic'].copy()
    history_gradCam_actor = history['gradCam_actor'].copy()
    history_gradCam_critic = history['gradCam_critic'].copy()
    history_gradCamGuided_actor = history['guidedGradCam_actor'].copy()
    history_gradCamGuided_critic = history['guidedGradCam_critic'].copy()
    fig_array[0,0] = normalization(history_gradients_actor, history, "gdb",GDB_actor=1)
    fig_array[0,1] = normalization(history_gradients_critic, history, 'gdb')
    fig_array[1,0] = normalization(history_gdb_actor, history, "gdb", GDB_actor=1)
    fig_array[1,1] = normalization(history_gdb_critic, history, 'gdb')
    fig_array[2,0] = normalization(history_gradCam_actor, history, "cam", )
    fig_array[2,1] = normalization(history_gradCam_critic, history, "cam")
    fig_array[3,0] = normalization(history_gradCam_actor, history, "cam", GDB_actor=1, guided_model=history_gdb_actor)
    fig_array[3,1] = normalization(history_gradCam_critic, history, 'cam',guided_model=history_gdb_critic)
    fig_array[4,0] = normalization(history_gradCamGuided_actor, history, "cam")
    fig_array[4,1] = normalization(history_gradCamGuided_critic, history, "cam")
    fig_array[5,0] = normalization(history_gradCamGuided_actor, history, "cam",GDB_actor=1, guided_model=history_gdb_actor)
    fig_array[5,1] = normalization(history_gradCamGuided_critic, history, 'cam',guided_model=history_gdb_critic)

    make_movie(args,history,fig_array,first_frame,num_frames,resolution,save_dir,prefix,env_name)





def normalization(heatmap, history, visu, GDB_actor=0, guided_model=None):
    frame=9
    heatmap=np.asarray(heatmap)
    guided_model=np.asarray(guided_model)
    if guided_model.all()==None:
        if visu=='gdb':
            print("normal")
            print(heatmap.shape)
            for i in range(heatmap.shape[0]):
                heatmap_ = heatmap[i,:,:,:,:]
                heatmap_-= heatmap_.mean() 
                heatmap[i,:,:,:,:]/= (heatmap_.std() + 1e-5) #
            if (GDB_actor):
                heatmap*=0.1#0.1
            else:
                heatmap*=0.1# 0.1 #0.1 
            print("d",heatmap.shape)
            heatmap = np.clip(heatmap, -1, 1)
            heatmap_pic1 = heatmap[:,0,:,:,frame]
        if visu=='cam':
            heatmap_pic1 = heatmap[:,:,:]
    else:
        print(" notnormal")
        print("ds",guided_model.shape)
        for i in range(guided_model.shape[0]):
            guided_model_ = guided_model[i,:,:,:,:]
            guided_model_-= guided_model_.mean() 
            guided_model[i,:,:,:,:]/= (guided_model_.std() + 1e-5) #
        if (GDB_actor):
            guided_model*=0.1#0.1
        else:
            guided_model*=0.1# 0.1
        guided_model = np.clip(guided_model, -1, 1)
        guided_model = guided_model[:,0,:,:,frame]
        guided_model[guided_model<0.0] = 0
        heatmap[heatmap<0.0] = 0
        heatmap_pic1 = (heatmap*guided_model)

    all_unproc_frames = history['un_proc_state'].copy()
    frame=np.zeros((num_frames,84,84,3))
    for i in range(len(all_unproc_frames)):
        frame[i,:,:,:]=np.asarray(Image.fromarray(all_unproc_frames[i]).resize((84, 84), Image.BILINEAR))/255
    proc_frame1 = overlap(frame,heatmap_pic1)
    
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

def make_movie(args,history,fig_array,first_frame,num_frames,resolution,save_dir,prefix,env_name ):
    movie_title ="{}-{}-{}.mp4".format(prefix, num_frames, env_name.lower())
    max_ep_len = first_frame + num_frames + 1
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='test', artist='mateus', comment='atari-video')
    writer = FFMpegWriter(fps=8, metadata=metadata)
    total_frames = len(history['state'])
    fig = plt.figure(figsize=[6, 6*1.3], dpi=resolution)
    print("fig_array.shape: ",fig_array.shape)
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
                    #ax.set_ylabel(titleListY[j])
                    #ax.set_xlabel(titleListX[k])

                    plt.imshow(img)
                    plt.axis('off')
                    z=z+1

            writer.grab_frame() 
            fig.clear()
            if i%100==0:
                print(i)



def play_game(args, agent, env, total_episodes=1):
    
    history = { 'state': [], 'un_proc_state' : [], 'action': [], 'gradients_actor':[], 'gradients_critic':[],'gradCam_actor':[],'gradCam_critic':[], 'gdb_actor':[],'gdb_critic':[], 'guidedGradCam_actor':[],'guidedGradCam_critic':[] ,'movie_frames':[]}
    rewards = []
    if agent.load_network:
        agent.q_network.load_weights(agent.load_network_path)
    for i in range(total_episodes):
        state = env.reset()
        #agent.init_game_setting()
        done = False
        episode_reward = 0.0
        action_state = agent.history_processor.process_state_for_network(
            agent.atari_processor.process_state_for_network(state))
        for _ in range(num_frames):
 
            action = agent.select_action(action_state, is_training=False)
            history['un_proc_state'].append(state)
            history['state'].append(action_state)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            history['action'].append(action)
            action_state = agent.history_processor.process_state_for_network(
                agent.atari_processor.process_state_for_network(state))
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
