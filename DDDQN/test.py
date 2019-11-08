import argparse
import numpy as np
from environment import Environment
import visualization.backpropagatio
import visualization.grad_cam.py


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


def make_video():
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
                gbp_heatmap_pic-= gbp_heatmap_pic.mean() #
                gbp_heatmap_pic/= (gbp_heatmap_pic.std() + 1e-5) #
                gbp_heatmap_pic*= 0.1 #
                frame=frame[0][0]

                # clip to [0, 1]
                gbp_heatmap_pic += 0.5
                gbp_heatmap_pic = np.clip(gbp_heatmap_pic, 0, 1)
                frame=frame/255
                frame=np.clip(frame,0,1)
                mixed = np.stack((gbp_heatmap_pic,gbp_heatmap_pic, frame), axis=2) 
                plt.imshow(mixed) ; 
                writer.grab_frame() ; f.clear()


def test(agent, env, total_episodes=30):
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