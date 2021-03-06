import argparse
from play_analyse import play_game, test
from environment import Environment


def parse():
    parser = argparse.ArgumentParser(description="MLDS&ADL HW3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')
    parser.add_argument('--gbp', action='store_false', help='visualize what the network learned with Guided backpropagation')
    parser.add_argument('--gradCAM', action='store_false', help='visualize what the network learned with GradCAM')
    parser.add_argument('--gbp_GradCAM', action='store_true', help='visualize what the network learned with Guided GradCAM')
    parser.add_argument('--visualize', action='store_true', help='visualize what the network learned with Guided GradCAM')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args


def run(args):
    # All frames are preprocessed with atari wrapper.
    if args.train_dqn:
        env_name = args.env_name or 'SeaquestNoFrameskip-v0'#'BreakoutNoFrameskip-v4'
        env = Environment(env_name, args, atari_wrapper=True)
        from agent_dir.agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        agent.train()

    if args.test_dqn:
        env_name = args.env_name or 'SeaquestNoFrameskip-v0'
        env = Environment(env_name, args, atari_wrapper=True, test=True, frame_stack_and_origin=True)
        from agent_dir.agent_dqn import Agent_DQN
        agent = Agent_DQN(env, args)
        
        if (args.visualize):
            print("<< visualization >>\n")
            play_game(args, agent, env, total_episodes=1)
        else:
            print("<< test >>\n")
            test(agent,env)


if __name__ == '__main__':
    import sys
    import numpy
    numpy.set_printoptions(threshold=sys.maxsize)
    args = parse()
    run(args)
