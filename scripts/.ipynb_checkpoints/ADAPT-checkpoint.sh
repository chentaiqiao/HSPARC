#!/bin/sh
env="StarCraft2"
algo="ADAPT"
exp="new"
seed=1
name="ADAPT"


map="3m"
ppo_epochs=10
ppo_clip=0.05
steps=10000
name="Ablation-CommFormer-v5-Post-01"
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1"
CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
  --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 2 --num_mini_batch 1 \
  --episode_length 100 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
  --use_value_active_masks --use_eval --prefix_name ${name} --use_bilevel --post_stable

# map="5m_vs_6m"
# ppo_epochs=10
# ppo_clip=0.05
# steps=10000000
# name="Ablation-CommFormer-v5-Post-01"
# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1"
# CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
#   --map_name ${map} --seed ${seed} --n_training_threads 16 --n_rollout_threads 32 --num_mini_batch 1 \
#   --episode_length 100 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
#   --use_value_active_masks --use_eval --prefix_name ${name} --use_bilevel --post_stable



# map="27m_vs_30m"
# ppo_epochs=5
# ppo_clip=0.2
# steps=10000000
# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1"
# CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
#   --map_name ${map} --seed ${seed} --n_training_threads 16 --n_rollout_threads 32 --num_mini_batch 1 \
#   --episode_length 100 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
#   --use_value_active_masks --use_eval --prefix_name ${name} --use_bilevel --post_stable --self_loop_add

# map="MMM2"
# ppo_epochs=10
# ppo_clip=0.05
# steps=10000000
# echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1"
# CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
#   --map_name ${map} --seed ${seed} --n_training_threads 16 --n_rollout_threads 32 --num_mini_batch 1 \
#   --episode_length 100 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
#   --use_value_active_masks --use_eval --prefix_name ${name} --use_bilevel --warmup 100


