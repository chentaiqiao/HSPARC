#!/bin/sh
env="magent"
scenario="adversarial_pursuit"  # or "battle"
map_size=45
n_agent=25  # for adversarial_pursuit: red team agents
algo="HSPARC"
exp="single"
seed=1
name="HSPARC"

ppo_epochs=5
ppo_clip=0.05
steps=5000000

echo "env is ${env}, scenario is ${scenario}, map_size is ${map_size}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=$1 python train/train_magent.py --seed ${seed} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario ${scenario} --map_size ${map_size} --n_agent ${n_agent} --lr 5e-4 --n_training_threads 1 \
    --n_rollout_threads 4 --num_mini_batch 1 --episode_length 400 --num_env_steps ${steps} --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} \
    --use_eval --use_value_active_masks --use_policy_active_masks \
    --prefix_name ${name} --use_bilevel --minimap_mode --global_reward

map_size=60
n_agent=45  # for adversarial_pursuit: red team agents
echo "env is ${env}, scenario is ${scenario}, map_size is ${map_size}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=$1 python train/train_magent.py --seed ${seed} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario ${scenario} --map_size ${map_size} --n_agent ${n_agent} --lr 5e-4 --n_training_threads 1 \
    --n_rollout_threads 4 --num_mini_batch 1 --episode_length 400 --num_env_steps ${steps} --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} \
    --use_eval --use_value_active_masks --use_policy_active_masks \
    --prefix_name ${name} --use_bilevel --minimap_mode --global_reward

map_size=70
n_agent=61  # for adversarial_pursuit: red team agents
echo "env is ${env}, scenario is ${scenario}, map_size is ${map_size}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=$1 python train/train_magent.py --seed ${seed} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario ${scenario} --map_size ${map_size} --n_agent ${n_agent} --lr 5e-4 --n_training_threads 1 \
    --n_rollout_threads 4 --num_mini_batch 1 --episode_length 400 --num_env_steps ${steps} --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} \
    --use_eval --use_value_active_masks --use_policy_active_masks \
    --prefix_name ${name} --use_bilevel --minimap_mode --global_reward

scenario="battle" 
map_size=25
n_agent=20 
echo "env is ${env}, scenario is ${scenario}, map_size is ${map_size}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=$1 python train/train_magent.py --seed ${seed} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario ${scenario} --map_size ${map_size} --n_agent ${n_agent} --lr 5e-4 --n_training_threads 1 \
    --n_rollout_threads 4 --num_mini_batch 1 --episode_length 400 --num_env_steps ${steps} --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} \
    --use_eval --use_value_active_masks --use_policy_active_masks \
    --prefix_name ${name} --use_bilevel --minimap_mode --global_reward

map_size=50
n_agent=100 
echo "env is ${env}, scenario is ${scenario}, map_size is ${map_size}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=$1 python train/train_magent.py --seed ${seed} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario ${scenario} --map_size ${map_size} --n_agent ${n_agent} --lr 5e-4 --n_training_threads 1 \
    --n_rollout_threads 4 --num_mini_batch 1 --episode_length 400 --num_env_steps ${steps} --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} \
    --use_eval --use_value_active_masks --use_policy_active_masks \
    --prefix_name ${name} --use_bilevel --minimap_mode --global_reward

map_size=80
n_agent=256 
echo "env is ${env}, scenario is ${scenario}, map_size is ${map_size}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=$1 python train/train_magent.py --seed ${seed} --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
    --scenario ${scenario} --map_size ${map_size} --n_agent ${n_agent} --lr 5e-4 --n_training_threads 1 \
    --n_rollout_threads 4 --num_mini_batch 1 --episode_length 400 --num_env_steps ${steps} --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} \
    --use_eval --use_value_active_masks --use_policy_active_masks \
    --prefix_name ${name} --use_bilevel --minimap_mode --global_reward