## Visual-Explanation-in-Deep-Reinforcement-Learning-through-Guided-Backpropagation-and-GradCam
This is the first algorithm which visualizes the knowledge of an agent trained by Deep Reinforcement Learning (paper will be published in Februar) using backpropagation. It shows why the agent is performing the action. Which pixels had the biggest influence on the decision of the agent.


# Deep Reinfocement Learning Algorithms:

[off Policy algorithms:]
- [X] DQN
- [x] DDQN
- [x] Dueling DDQN
- [ ] LSTM DQN
- [ ] Bidirectional LSTM DQN
- [ ] Attention LSTM DQN

[on Policy algorithms:]
- [ ] Policy Gradient
- [ ] A3C

# Visualization Techniques:

- [X] Backpropagation
- [x] Guided Backpropagation
- [ ] GradCam
- [ ] Guided GradCam
- [ ] SmoothGrad
- [ ] Perturbation-based Saliency Map




For now we will compare how good we can visualize DQN and dueling double DQN Algorithms. As environment we will use ATARI Game and later Doom.


If you have a trained dueling agent and you want visualize what he learned run:

```console
python3 main.py --test_dqn --gbp --dueling True --ddqn True --test_dqn_model_path saved_dqn_networks/XXXX.h5
```
otherwise train you agent with 

```console
python3 main.py --train_dqn --dueling True --ddqn True
```
and you can test how the agent plays with:

```console
python3 main.py --test_dqn --do_render --dueling True --ddqn True --test_dqn_model_path saved_dqn_networks/XXXX.h5
```

![Alt text](pictures/DuelingNet.png?raw=true "DQN vs. Dueling DQN Network")

# some results
Here we can see how the agent is looking more on his position (in the advantage part of the neuronal network [right figure] and how he is looking more on the reward in the value function part of the network [left figure].
(left value || right advantage)
![Alt text](pictures/4.png?raw=true "example with environment")
![Alt text](pictures/1.png?raw=true "example 1")
![Alt text](pictures/2.png?raw=true "example 2")

For Attention Networks (Visualization is not working yet):

train:
```console
python3 dqn_atari.py --net_mode duel --ddqn --num_frames 10 --no_monitor --selector --bidir --recurrent --a_t --selector --task_name 'SpatialAt_DQN' --seed 36 --env 'SeaquestDeterministic-v0'
```
test:
```console
python3 dqn_atari.py --net_mode duel --ddqn --num_frames 10 --recurrent --a_t --bidir --selector --task_name 'SpatialAt_DQN' --test --load_network --test --load_network_path=log/SeaquestDeterministic-v0-run6-SpatialAt_DQN/qnet7628.h5 --env 'SeaquestDeterministic-v0'
```
