# Visual-Explanation-in-Deep-Reinforcement-Learning-through-Guided-Backpropagation-and-GradCam
This is the first algorithm which visualizes the knowledge of an agent trained by Deep Reinforcement Learning (paper will be published in Februar) using backpropagation. It shows why the agent is performing the action. Which pixels had the biggest influence on the decision of the agent.


This code will include in the futer not only visual explanation through backpropagation but also through grad CAM.

For now we will compare how good we can visualize DQN and dueling double DQN Algorithmes. As environment we will use ATARI Game and later Doom.


If you have a trained dueling agent run:

```console
python3 main.py --test_dqn --gbp --dueling True --ddqn True --test_dqn_model_path saved_dqn_networks/XXXX.h5```

otherwise train you agent with 

```console
python3 main.py --train_dqn --dueling True --ddqn True```

and you can test how the agent plays with:

```console
python3 main.py --test_dqn --do_render --dueling True --ddqn True --test_dqn_model_path saved_dqn_networks/XXXX.h5```
