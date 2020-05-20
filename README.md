## Visual-Explanation-in-Deep-Reinforcement-Learning-through-Guided-Backpropagation-and-GradCam
Deep Reinforcement Learning (DRL) connects the classic Reinforcement Learning algorithms with Convolutional Neural Networks (CNN). A problem in DRL is that CNNs are black-boxes and it is not easy to understand the decision-making process of agents. In order to be able to use the programs in highly dangerous environments for humans and machines, the developer needs a debugging tool to assure that the program does what is expected. Currently, the rewards are primarily used ever, this can lead to deceptive fallacies if policy and not learning to respond to the problem can be recognized with the help of to interpret how well an agent is learning. How the agent receives more rewards by memorizing a environment. In this work, it is shown that this visualization techniques.

This work brings some of the best-known visualization methods from the field of image classification to the area of Deep Reinforcement Learning (DRL). Furthermore, two new visualization
techniques have been developed, one of which provides particularly good results.
It is being proven to what extent the algorithms can be used in the area of Reinforcement learning. Also, the question arises on how well the different DRL algorithms can be visualized in
different environments from different visualization techniques.

Among other things, the results of this work refute the claims made by Sam Greydanus et
al. ("Visualizing and Understanding Atari Agents", 2017) that guided backpropagation cannot
be used for visualization techniques. Furthermore, the assertion made by Julius Adebayo et al.
("Sanity Checks for Saliency Maps", 2018), that guided backpropagation and guided Grad-Cam
(at least in image processing) do not visualize the learned model but work similarly to an edge
detector do not apply to deep reinforcement learning as it is shown in this work.
However, since the results of the visualization techniques strongly depend on the quality of the
neural network, one new architecture for off-policy algorithms was also developed in this work.
The structure of the developed Networks surpasses the DDDQN and Attention DRQN Networks.

Finally, a theoretical elaboration on the function and significance of bidirectional neural networks for deep reinforcement learning was developed. It is known that long short-term memory
(LSTM) layers transport information like speed, acceleration, trajectories, etc. of features.
Nonetheless, there is no elaboration on which information bi-directional networks transport in
the area of deep reinforcement learning and how this information affects learning behaviour. If
LSTM layers evaluate the current state based on past information, how can the bi-directional
network evaluate the current state based on future information if this has not yet happened? Furthermore, this hypothesis claims why off-policy algorithms could behave similarly to on-policy
algorithms by bidirectional LSTM networks

# Deep Reinfocement Learning Algorithms:

[off Policy algorithms:]
- [X] DQN
- [x] DDQN
- [x] Dueling DDQN
- [X] LSTM D/D/DQN
- [X] Bidirectional LSTM DQN
- [X] Attention LSTM DQN
- [X] Splitted Attention LSTM DQN

[on Policy algorithms:]
- [X] A3C
- [X] A3C Attention with LSTM

# Visualization Techniques:

- [X] Backpropagation
- [x] Guided Backpropagation
- [X] GradCam
- [X] Guided GradCam (multiplication of Guided Back. and GradCam)
- [X] G1GradCam (GradCam with Guided Model)
- [X] G2GradCam (GradCam with Guided Model and multipied with Guided Back.)

## Splitted Attention DDQN
I Introduce a new model of an off-policy network which performes better then the state of the art Attention DDDQN (In average this Splitted Attention DDQN Network is making 7-8000 points in the game Seaquest-v0. DDDQN with Attation what I take as Basline the Network made 3-4000 Points). 
This Network has in the input time distributed convolutional layers after that we splitt the layers into an advatage and a value stream. On the bottom of both we have bidriectional LSTM networks followed with an attention network. On the end we bring both streams together like in the dueling Network.

![Alt text](splitted_attention_DDDQN/Master_Network/Attention-DQN_duel_visual_improved_02/model_plot.png)








For now we will compare how good we can visualize DQN and dueling double DQN Algorithms. As environment we will use ATARI.


If you have a trained dueling agent and you want visualize what he learned run:

```console
python3 main.py --test_dqn --gbp --dueling True --ddqn True --test_dqn_model_path saved_dqn_networks/XXXX.h5
```
otherwise train your agent with 

```console
python3 main.py --train_dqn --dueling True --ddqn True
```
and you can test how the agent plays with:

```console
python3 main.py --train_dqn --test_dqn --dueling True --visualize --ddqn True --test_dqn_model_path seqquest_dqn_5300000.h5 --env BreakoutNoFrameskip-v0
```


## some results

# A3C
In this GIF you can see the different visualizations for the A3C algorithm Playing Breakout-v0 with 3 frames:
![](A3C/a3c_vanila/movies/450_breakout.gif)

# DDDQN
Here we can see how the agent is looking more on his position (in the advantage part of the neuronal network [right figure] and how he is looking more on the reward in the value function part of the network [left figure].
(left value || right advantage)
![Alt text](pictures/4.png?raw=true "example with environment")


For Attention Networks:

train:
```console
python3 dqn_atari.py --net_mode duel --ddqn --num_frames 10 --no_monitor --selector --bidir --recurrent --a_t --selector --task_name 'SpatialAt_DQN' --seed 36 --env 'SeaquestDeterministic-v0'
```
test:
```console
python3 dqn_atari.py --net_mode duel_at_improved --ddqn --num_frames 10 --recurrent --a_t --bidir  --test --load_network --load_network_path 0_Best_4956.h5 --env 'Seaquest-v0' 
```
