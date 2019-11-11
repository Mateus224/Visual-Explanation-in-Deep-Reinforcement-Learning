import argparse
import numpy as np
from environment import Environment
import matplotlib.animation as manimation
import matplotlib.pyplot as plt

from visualization.backpropagation import *
#import visualization.grad_cam.py


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


def make_movie(args, agent, history, first_frame=0, num_frames=100, prefix='default', resolution=75, save_dir='./movies/', env_name='Breakout-v0'):
    visualization_network_model = build_guided_model(agent)
    visualization_network_model.load_weights(args.test_dqn_model_path)
    movie_title ="{}-{}-{}.mp4".format(prefix, num_frames, env_name.lower())
    max_ep_len = first_frame + num_frames + 1
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='test', artist='mateus', comment='atari-video')
    writer = FFMpegWriter(fps=8, metadata=metadata)
    total_frames = len(history['state'])
    frame_1= np.zeros((84, 84))
    f = plt.figure(figsize=[6, 6*1.3], dpi=resolution)
    with writer.saving(f, save_dir + movie_title, resolution):
        for i in range(num_frames):
            ix = first_frame+i
            if ix < total_frames: # prevent loop from trying to process a frame ix greater than rollout length
                frame = history['state'][ix].copy()
                frame = np.expand_dims(frame, axis=0)
                print(ix)
                action = history['action'][ix].copy()
                gbp_heatmap = guided_backprop(args, visualization_network_model,"dense_10", frame)
                #print(gbp_heatmap.shape,"headmap")
                #geb_gradCam = grad_cam.grad_cam(model,"dense_2",frame, action) 
                gbp_heatmap_pic=gbp_heatmap[0,:,:,0]

                gbp_heatmap_pic-= gbp_heatmap_pic.mean() #
                gbp_heatmap_pic/= (gbp_heatmap_pic.std() + 1e-5) #
                gbp_heatmap_pic*= 0.1 #
                #print(frame.shape, "jas")
                frame=frame[0,:,:,0]
                print(frame.shape)
                #print(gbp_heatmap_pic.shape, "asdasdasd")
                #frame_1=frame[:,:,0]
                #print(frame_1.shape)

                # clip to [0, 1]
                gbp_heatmap_pic += 0.5
                gbp_heatmap_pic = np.clip(gbp_heatmap_pic, 0, 1)
                frame=frame/255
                frame=np.clip(frame,0,1)
                mixed = np.stack((gbp_heatmap_pic,gbp_heatmap_pic, frame), axis=2) 
                plt.imshow(mixed) 
                writer.grab_frame() 
                f.clear()


def test(args, agent, env, total_episodes=30):
    if args.gbp or args.gradCAM or args.gbp_GradCAM:
        history = { 'state': [], 'action': []}
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
            if args.gbp or args.gradCAM or args.gbp_GradCAM:
                history['state'].append(state)
                history['action'].append(action)
        rewards.append(episode_reward)
    print('Run %d episodes'%(total_episodes))
    print('Mean:', np.mean(rewards))
    if args.gbp or args.gradCAM or args.gbp_GradCAM:
        make_movie(args, agent, history)

    return history


def run(args):
    if args.test_dqn:
        env = Environment('BreakoutNoFrameskip-v4', args, atari_wrapper=True, test=True)
        from agent_dir.agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        history=test(agent, env, total_episodes=100)
        make_video(history)






if __name__ == '__main__':
    args = parse()
    run(args)