## I Introduce A new Architect
I am presenting a new architecture of neural networks for off-policy algorithms. This structure doubles the points in the game Seaquest-v0 where the game has been tested.



If you have a trained dueling agent and you want visualize what he learned run:

```console
python3 dqn_atari.py   --net_mode duel_at_improved --ddqn --num_frames 10  --recurrent --duel_visual --gbp --bidir --load_network --load_network_path ../log/XXX --env 'Seaquest-v0'
```
otherwise train you agent with 

```console
python3 dqn_atari.py   --net_mode duel_at_improved --ddqn --num_frames 10  --recurrent --bidir --env 'Seaquest-v0'
```
