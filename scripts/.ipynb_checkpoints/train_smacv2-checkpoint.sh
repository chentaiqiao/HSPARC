#!/bin/sh
env="StarCraft2v2"
algo="HSPARC"
map="10gen_terran"
exp="single"
seed=1
name="HSPARC"
steps=10010000
ppo_epochs=10
ppo_clip=0.05

units="10v11"
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1, unit is ${units}"
CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
  --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 4 --num_mini_batch 1 \
  --episode_length 400 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
  --use_value_active_masks --use_eval --prefix_name ${name} --use_recurrent_policy --b_agent 3 --num_env_steps ${steps} --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip}

units="20v20"
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1, unit is ${units}"
CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
  --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 4 --num_mini_batch 1 \
  --episode_length 400 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
  --use_value_active_masks --use_eval --prefix_name ${name} --use_recurrent_policy --b_agent 3 

map="10gen_protoss"


units="10v11"
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1, unit is ${units}"
CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
  --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 4 --num_mini_batch 1 \
  --episode_length 400 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
  --use_value_active_masks --use_eval --prefix_name ${name} --use_recurrent_policy --b_agent 3

units="20v20"
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1, unit is ${units}"
CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
  --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 4 --num_mini_batch 1 \
  --episode_length 400 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
  --use_value_active_masks --use_eval --prefix_name ${name} --use_recurrent_policy --b_agent 3 

map="10gen_zerg"


units="10v11"
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1, unit is ${units}"
CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
  --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 4 --num_mini_batch 1 \
  --episode_length 400 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
  --use_value_active_masks --use_eval --prefix_name ${name} --use_recurrent_policy --b_agent 3

units="20v20"
echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cuda is $1, unit is ${units}"
CUDA_VISIBLE_DEVICES=$1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} \
  --map_name ${map} --seed ${seed} --n_training_threads 1 --n_rollout_threads 4 --num_mini_batch 1 \
  --episode_length 400 --num_env_steps ${steps} --lr 5e-4 --ppo_epoch ${ppo_epochs} --clip_param ${ppo_clip} --save_interval 100000 \
  --use_value_active_masks --use_eval --prefix_name ${name} --use_recurrent_policy --b_agent 3 


