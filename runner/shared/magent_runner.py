# HSPARC/runner/shared/magent_runner.py
import time
import wandb
import numpy as np
from functools import reduce
import torch
from HSPARC.runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class MAgentRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for MAgent environments."""
    
    def __init__(self, config):
        super(MAgentRunner, self).__init__(config)
        
        # MAgent specific metrics
        self.red_team_wins = 0
        self.blue_team_wins = 0
        self.episode_steps = []

    def run(self):
        """Main training loop for MAgent environments."""
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        # Training metrics
        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        done_episodes_rewards = []
        
        # MAgent specific metrics
        train_episode_red_alive = [0 for _ in range(self.n_rollout_threads)]
        train_episode_blue_alive = [0 for _ in range(self.n_rollout_threads)]
        done_episodes_red_alive = []
        done_episodes_blue_alive = []
        done_episodes_steps = []

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                # Observe reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                # Process environment information
                dones_env = np.all(dones, axis=1)
                reward_env = np.mean(rewards, axis=1).flatten()
                train_episode_rewards += reward_env

                # Extract MAgent specific metrics
                red_alive = [t_info[0].get("red_team_alives", 0) for t_info in infos]
                blue_alive = [t_info[0].get("blue_team_alives", 0) for t_info in infos]
                
                train_episode_red_alive = [r + ra for r, ra in zip(train_episode_red_alive, red_alive)]
                train_episode_blue_alive = [b + ba for b, ba in zip(train_episode_blue_alive, blue_alive)]

                # Track episode completion
                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        # Record completed episode metrics
                        done_episodes_rewards.append(train_episode_rewards[t])
                        done_episodes_red_alive.append(train_episode_red_alive[t] / (step + 1))  # Average alive count
                        done_episodes_blue_alive.append(train_episode_blue_alive[t] / (step + 1))
                        done_episodes_steps.append(step + 1)
                        
                        # Track team wins
                        if infos[t][0].get("red_team_win", False):
                            self.red_team_wins += 1
                        elif infos[t][0].get("blue_team_win", False):
                            self.blue_team_wins += 1
                        
                        # Reset thread metrics
                        train_episode_rewards[t] = 0
                        train_episode_red_alive[t] = 0
                        train_episode_blue_alive[t] = 0

                # Prepare data for buffer insertion
                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic

                # Insert data into buffer
                self.insert(data)

            # Compute returns and update network
            self.compute()
            train_infos = self.train()

            # Post-process and logging
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # Save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save(episode)

            # Log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n MAgent Environment {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.env_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                # Log training information
                self.log_train(train_infos, total_num_steps)

                # Log MAgent specific metrics
                if len(done_episodes_rewards) > 0:
                    # Episode rewards and survival metrics
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    aver_episode_red_alive = np.mean(done_episodes_red_alive)
                    aver_episode_blue_alive = np.mean(done_episodes_blue_alive)
                    aver_episode_steps = np.mean(done_episodes_steps)
                    
                    # Win rates
                    total_wins = self.red_team_wins + self.blue_team_wins
                    red_win_rate = self.red_team_wins / total_wins if total_wins > 0 else 0
                    blue_win_rate = self.blue_team_wins / total_wins if total_wins > 0 else 0
                    
                    # Log to tensorboard/wandb
                    magent_metrics = {
                        "magent/average_episode_rewards": aver_episode_rewards,
                        "magent/average_episode_steps": aver_episode_steps,
                        "magent/average_red_alive": aver_episode_red_alive,
                        "magent/average_blue_alive": aver_episode_blue_alive,
                        "magent/red_win_rate": red_win_rate,
                        "magent/blue_win_rate": blue_win_rate,
                        "magent/red_team_wins": self.red_team_wins,
                        "magent/blue_team_wins": self.blue_team_wins
                    }
                    
                    for k, v in magent_metrics.items():
                        if self.use_wandb:
                            wandb.log({k: v}, step=total_num_steps)
                        else:
                            self.writter.add_scalar(k, v, total_num_steps)
                    
                    print("MAgent Metrics - Avg Reward: {:.2f}, Steps: {:.1f}, Red Alive: {:.1f}, Blue Alive: {:.1f}, Red Win Rate: {:.3f}"
                          .format(aver_episode_rewards, aver_episode_steps, aver_episode_red_alive, 
                                 aver_episode_blue_alive, red_win_rate))
                    
                    # Reset episode metrics
                    done_episodes_rewards = []
                    done_episodes_red_alive = []
                    done_episodes_blue_alive = []
                    done_episodes_steps = []

            # Evaluation
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        """Reset environment and initialize buffer."""
        obs, share_obs, available_actions = self.envs.reset()

        # Use centralized V or not
        if not self.use_centralized_V:
            share_obs = obs

        # Initialize buffer
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        """Collect actions from policy."""
        self.trainer.prep_rollout()
        
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]),
                                              np.concatenate(self.buffer.available_actions[step]),
                                              scoring_network=self.trainer.scoring_network)
        
        # Split results by environment threads
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        """Insert collected data into replay buffer."""
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        # Reset RNN states for done environments
        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        # Create masks
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # Use centralized V or not
        if not self.use_centralized_V:
            share_obs = obs

        # Insert into buffer
        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, None, active_masks,
                           available_actions)

    def log_train(self, train_infos, total_num_steps):
        """Log training information."""
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        print("average_step_rewards is {}.".format(train_infos["average_step_rewards"]))
        
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        """Evaluate policy performance."""
        eval_episode = 0
        eval_episode_rewards = []
        eval_episode_red_alive = []
        eval_episode_blue_alive = []
        eval_episode_steps = []
        
        one_episode_rewards = [0 for _ in range(self.n_eval_rollout_threads)]
        one_episode_red_alive = [0 for _ in range(self.n_eval_rollout_threads)]
        one_episode_blue_alive = [0 for _ in range(self.n_eval_rollout_threads)]
        one_episode_steps = [0 for _ in range(self.n_eval_rollout_threads)]

        # Reset evaluation environment
        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N,
                                    self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_share_obs),
                                        np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(eval_available_actions),
                                        deterministic=True)
            
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # Step environment
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)

            # Process rewards and metrics
            eval_rewards_mean = np.mean(eval_rewards, axis=1).flatten()
            one_episode_rewards = [r + rew for r, rew in zip(one_episode_rewards, eval_rewards_mean)]
            
            # Extract MAgent metrics
            red_alive = [t_info[0].get("red_team_alives", 0) for t_info in eval_infos]
            blue_alive = [t_info[0].get("blue_team_alives", 0) for t_info in eval_infos]
            
            one_episode_red_alive = [r + ra for r, ra in zip(one_episode_red_alive, red_alive)]
            one_episode_blue_alive = [b + ba for b, ba in zip(one_episode_blue_alive, blue_alive)]
            one_episode_steps = [s + 1 for s in one_episode_steps]

            # Handle episode completion
            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents,
                                                                self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                          dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    
                    # Record episode metrics
                    eval_episode_rewards.append(one_episode_rewards[eval_i])
                    eval_episode_red_alive.append(one_episode_red_alive[eval_i] / one_episode_steps[eval_i])
                    eval_episode_blue_alive.append(one_episode_blue_alive[eval_i] / one_episode_steps[eval_i])
                    eval_episode_steps.append(one_episode_steps[eval_i])
                    
                    # Reset thread metrics
                    one_episode_rewards[eval_i] = 0
                    one_episode_red_alive[eval_i] = 0
                    one_episode_blue_alive[eval_i] = 0
                    one_episode_steps[eval_i] = 0

            # Check if evaluation is complete
            if eval_episode >= self.all_args.eval_episodes:
                # Log evaluation results
                eval_metrics = {
                    'eval/average_episode_rewards': np.mean(eval_episode_rewards),
                    'eval/average_episode_steps': np.mean(eval_episode_steps),
                    'eval/average_red_alive': np.mean(eval_episode_red_alive),
                    'eval/average_blue_alive': np.mean(eval_episode_blue_alive),
                    'eval/max_episode_rewards': np.max(eval_episode_rewards),
                    'eval/min_episode_rewards': np.min(eval_episode_rewards)
                }
                
                self.log_env(eval_metrics, total_num_steps)
                
                print("Evaluation Results - Avg Reward: {:.2f}, Steps: {:.1f}, Red Alive: {:.1f}, Blue Alive: {:.1f}"
                      .format(np.mean(eval_episode_rewards), np.mean(eval_episode_steps),
                             np.mean(eval_episode_red_alive), np.mean(eval_episode_blue_alive)))
                break