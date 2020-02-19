import argparse
import numpy as np
#from environment import Environment
import matplotlib.animation as manimation
import matplotlib.pyplot as plt


from visualization.backpropagation import *
from PIL import Image
from visualization.grad_cam import *
from visualization.model import build_network
from scipy.misc.pilutil import imresize
from scipy.misc.pilutil import imread
from skimage.color import rgb2gray
#import visualization.grad_cam.py

num_frames=80


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


def init_saliency_map(args, agent, history, first_frame=0, prefix='QF_', resolution=300, save_dir='./movies/', env_name="DDDQN"):

    load_model = build_network([agent.frame_width, agent.frame_height, agent.state_length], agent.num_actions)
    #_, _,load_model_cam,_ = build_network(agent.observation_shape, agent.action_space_n)
    #_, _, load_model_guided_backprop ,_= build_guided_model(agent.observation_shape, agent.action_space_n)
    load_guided_model = build_guided_model([agent.frame_width, agent.frame_height, agent.state_length], agent.num_actions)

    load_model.load_weights(args.test_dqn_model_path)
    load_guided_model.load_weights(args.test_dqn_model_path)
    #load_model_guided_backprop.load_weights(args.load_network_path)
    #load_model_grad_cam.load_weights(args.load_network_path)
    frame_1= np.zeros((84, 84))
    total_frames=len(history['state'])
    backprop_actor = init_guided_backprop(load_model,"dense_12")
    backprop_critic = init_guided_backprop(load_model,"dense_12")
    cam_actor = init_grad_cam(load_model, "convolution2d_7")
    cam_critic = init_grad_cam(load_model, "convolution2d_7")#, False)
    guidedBackprop_actor = init_guided_backprop(load_guided_model,"dense_16")
    guidedBackprop_critic = init_guided_backprop(load_guided_model,"dense_16")
    gradCAM_actor = init_grad_cam(load_guided_model, "convolution2d_10")
    gradCAM_critic = init_grad_cam(load_guided_model, "convolution2d_10")#, False)
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

            gradCam_heatmap = grad_cam(cam_critic, frame, action)
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

            gradCam_heatmap = grad_cam(gradCAM_critic, frame, action)
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

#def normalization_cam(cam_heatmap,history)


#def save_for_GradCam(heatmap, gdp=1)



def normalization(heatmap, history, visu, GDB_actor=0, guided_model=None):
    frame=0
    heatmap=np.asarray(heatmap)
    guided_model=np.asarray(guided_model)
    if guided_model.all()==None:
        if visu=='gdb':
            print("normal")
            heatmap = heatmap[:,:,:]
            #gbp_heatmap_pic=gbp_heatmap[0,:,:,:]
            heatmap-= heatmap.mean() 
            heatmap/= (heatmap.std() + 1e-5) #
            if (GDB_actor):
                #print(heatmap)
                heatmap*=0.1
            else:
                heatmap*= 0.1 
            heatmap = np.clip(heatmap, -1, 1)
            heatmap_pic1 = heatmap[:,0,:,:,frame]
        if visu=='cam':
            heatmap_pic1 = heatmap[:,:,:]
    else:
        print(" notnormal")
        guided_model = guided_model[:,:,:]
        #gbp_heatmap_pic=gbp_heatmap[0,:,:,:]
        guided_model-= guided_model.mean() 
        guided_model/= (guided_model.std() + 1e-5) #
        if (GDB_actor):
            #print(heatmap)
            guided_model*=0.1
        else:
            guided_model*= 0.1 #0.1 
        guided_model = np.clip(guided_model, -1, 1)
        guided_model = guided_model[:,0,:,:,frame]
        guided_model[guided_model<0.0] = 0
        heatmap[heatmap<0.0] = 0
        heatmap_pic1 = (heatmap*guided_model)
        heatmap_pic1[heatmap_pic1<0.0]=0

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
    _gbp_heatmap = [gbp_heatmap for _ in range(3)]
    _gbp_heatmap=np.stack(_gbp_heatmap,axis=3)
    gbp_heatmap=_gbp_heatmap
    gbp_heatmap_pos=np.asarray(gbp_heatmap.copy())
    gbp_heatmap_neg=np.asarray(gbp_heatmap.copy())
    gbp_heatmap_pos[gbp_heatmap_pos<0.0]=0
    gbp_heatmap_neg[gbp_heatmap_neg>=0.0]=0
    gbp_heatmap_neg=-gbp_heatmap_neg
    gbp_heatmap = color_pos * gbp_heatmap_pos[:,:,:,:] + color_neg * gbp_heatmap_neg[:,:,:,:]
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
                    z=z+1

            writer.grab_frame() 
            fig.clear()
            if i%100==0:
                print(i)



def play_game(args, agent, env, total_episodes=1):
    
    history = { 'state': [], 'un_proc_state' : [], 'action': [], 'gradients_actor':[], 'gradients_critic':[],'gradCam_actor':[],'gradCam_critic':[], 'gdb_actor':[],'gdb_critic':[], 'guidedGradCam_actor':[],'guidedGradCam_critic':[] ,'movie_frames':[]}
    rewards = []
    for i in range(total_episodes):
        state, origin_state = env.reset()
        #print("state:",state.shape)
        #print("orgin_state:",origin_state.shape)
        #prozess_atari_wraper_frames(origin_state, state)
        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        for _ in range(num_frames):
            state = prozess_atari_wraper_frames(state=state)
            history['state'].append(state)
            origin_state=(origin_state * 255).astype(np.uint8)
            history['un_proc_state'].append(origin_state)
            action = agent.make_action(state, test=True)
            state,origin_state, reward, done, info = env.step(action)
            episode_reward += reward
            #agent.save_observation(state)
            #action_state=agent.observations
            history["action"].append(action)
        rewards.append(episode_reward)
    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))
    init_saliency_map(args, agent, history)

    return history

def save_observation(observations):
    observations = np.roll(observations, -1, axis=0)
    observation=np.zeros([84,84,4])
    observation[:,:,0]= transform_screen(observations[0,:,:,:])
    observation[:,:,1]= transform_screen(observations[1,:,:,:])
    observation[:,:,2]= transform_screen(observations[2,:,:,:])
    observation[:,:,3]= transform_screen(observations[3,:,:,:])
    return observation
    
def transform_screen(data):
    return rgb2gray(imresize(data, [84,84]))[None, ...]


def prozess_atari_wraper_frames(origin_state=None, state=None):
    new_frame=state.reshape(210,160,4,3)
    new_frame=np.moveaxis(new_frame, 0,-2)
    new_frame=np.moveaxis(new_frame, 0,-2)
    new_frame = save_observation(new_frame)
    return(new_frame)
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='test', artist='mateus', comment='atari-video')
    writer = FFMpegWriter(fps=8)
    total_frames = 4
    fig = plt.figure(figsize=[6, 6*1.3], dpi=75)
    with writer.saving(fig, "test.mp4", 75):
        for i in range(4):#total_frames): #num_frames

            img = new_frame[i,:,:,:]
            plt.imshow(img)

            writer.grab_frame() 
            fig.clear()
            if i%100==0:
                print(i)


def test(agent, env, total_episodes=30):
    rewards = []
    for i in range(total_episodes):
        state, origin_state = env.reset()
        
        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        #playing one game
        while(not done):
            action = agent.make_action(prozess_atari_wraper_frames(state=state), test=True)
            state, orgin_state, reward, done, info = env.step(action)
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
