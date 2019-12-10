#!/bin/bash
#SBATCH --nodelist=hpc-dnode3
#SBATCH --job-name=SeaQuest_dueling_a_t_10_frames
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=matthias.rosynski@dfki.de
#SBATCH --partition=gpu_volta
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=18G
#SBATCH -o /gluster/home/mrosynski/SCRATCH/output/seaquest/a_t_10_Deterministic.out

source /gluster/home/mrosynski/SCRATCH/py_env/bin/activate
python3 ./scripts/Attention-DQN/dqn_atari.py --net_mode duel --ddqn --num_frames 10 --no_monitor --selector --bidir --recurrent --a_t --selector --task_name 'SpatialAt_DQN' --seed 36 --env 'SeaquestDeterministic-v0'


