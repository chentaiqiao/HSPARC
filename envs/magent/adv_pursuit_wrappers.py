import math
from statistics import mean
import supersuit as ss
import numpy as np
import torch as th
import gym
from gym.spaces import Dict as GymDict, Box, Discrete 
import pettingzoo.magent as magent

from ...pretrained.magent import IDQN_AdvPursuit
from ..multiagentenv import MultiAgentEnv
from .magent_env import PettingZooEnv

REGISTRY = {}
MAPSIZE2N = {
    45: (25, 50), 50: (31, 62), 55: (37, 75), 60: (45, 90), 
    65: (52, 105), 70: (61, 122)
}

class AdvPursuitEnv(MultiAgentEnv):
    
    def __init__(self, **kwargs):
        super().__init__()
        
        self.map_name = kwargs["env_args"].get("map_name", "adversarial_pursuit_v3")
        self.max_cycles = kwargs["env_args"].get("max_cycles", 200)
        self.seed = kwargs["env_args"].get("seed", None)
        
        map_size = int(kwargs["env_args"].get("map_size", 45))
        env = magent.adversarial_pursuit_v3.parallel_env(
            map_size=map_size,
            max_cycles=self.max_cycles,
            minimap_mode=kwargs["env_args"].get("minimap_mode", True)
        )
        
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
        
        self.agents = env.possible_agents
        
        self.red_prefix = 'red'
        self.blue_prefix = 'blue'
        if any(a.startswith('predator') for a in self.agents):
            self.red_prefix = 'predator'
            self.blue_prefix = 'prey'
            
        self.n_agents = len([a for a in self.agents if a.startswith(self.red_prefix)])
        self.n_preys = len([a for a in self.agents if a.startswith(self.blue_prefix)])
        
        if self.n_agents == 0:
            raise ValueError(f"No controlled agents found. Check map_size ({map_size}) or agent naming.")

        self.episode_limit = self.max_cycles
        
        self.env = PettingZooEnv(env)
        
        obs_sample = self.env.reset()
        
        self.obs_shape = (0,)
        if self.n_agents > 0:
            first_red = next((a for a in self.agents if a.startswith(self.red_prefix)), None)
            if first_red:
                if isinstance(obs_sample, dict) and first_red in obs_sample:
                    first_obs = obs_sample[first_red]
                    self.obs_shape = (np.prod(first_obs.shape),)
                elif isinstance(obs_sample, dict) and len(obs_sample) > 0:
                    first_obs = list(obs_sample.values())[0]
                    self.obs_shape = (np.prod(first_obs.shape),)
        
        self.observation_space = [Box(low=-np.inf, high=np.inf, shape=self.obs_shape) 
                                for _ in range(self.n_agents)]
        
        state_dim = self.obs_shape[0] * self.n_agents
        self.share_observation_space = [Box(low=-np.inf, high=np.inf, shape=(state_dim,)) 
                                for _ in range(self.n_agents)]
        
        try:
            n_actions = self.env.action_space.n
        except AttributeError:
            if hasattr(env, 'action_space'):
                act_space = env.action_space(self.agents[0])
                n_actions = act_space.n
            else:
                n_actions = 13

        self.action_space = [gym.spaces.Discrete(n_actions) 
                           for _ in range(self.n_agents)]

    def reset(self):
        obs_dict = self.env.reset()
        obs = []
        for agent in self.agents:
            if agent.startswith(self.red_prefix):
                if agent in obs_dict:
                    obs.append(obs_dict[agent].flatten())
                else:
                    obs.append(np.zeros(self.obs_shape))
        
        state = self._get_state(obs_dict)
        avail_actions = self._get_avail_actions()
        
        return obs, state, avail_actions

    def step(self, actions):
        action_dict = {}
        red_agents = [a for a in self.agents if a.startswith(self.red_prefix)]
        blue_agents = [a for a in self.agents if a.startswith(self.blue_prefix)]
        
        for i, agent in enumerate(red_agents):
            if i < len(actions):
                action_dict[agent] = int(actions[i])
        
        for agent in blue_agents:
            try:
                if hasattr(self.env, 'agents') and agent not in self.env.agents:
                    continue
                action_dict[agent] = self.env.action_space.sample()
            except:
                pass
        
        obs_dict, rewards, dones, raw_infos = self.env.step(action_dict)
        
        obs = []
        red_rewards = []
        for agent in red_agents:
            if agent in obs_dict:
                obs.append(obs_dict[agent].flatten())
                red_rewards.append(rewards.get(agent, 0.0))
            else:
                obs.append(np.zeros(self.obs_shape))
                red_rewards.append(0.0)
        
        state = self._get_state(obs_dict)
        avail_actions = self._get_avail_actions()
        
        if isinstance(dones, dict):
            done_env = dones.get("__all__", False)
        else:
            done_env = dones
        

        current_agents = self.env.agents
        red_alive = len([a for a in current_agents if a.startswith(self.red_prefix)])
        blue_alive = len([a for a in current_agents if a.startswith(self.blue_prefix)])
        

        red_win = False
        blue_win = False
        if done_env:
            if blue_alive == 0:
                red_win = True  
            else:
                blue_win = True 

        final_infos = []
        for i, agent in enumerate(red_agents):

            agent_info = raw_infos.get(agent, {}).copy()
            

            if i == 0:
                agent_info['red_team_alives'] = red_alive
                agent_info['blue_team_alives'] = blue_alive
                agent_info['red_team_win'] = red_win
                agent_info['blue_team_win'] = blue_win
            
            final_infos.append(agent_info)
            
        rewards = [[r] for r in red_rewards]
        dones_list = [done_env] * self.n_agents
        
        return obs, state, rewards, dones_list, final_infos, avail_actions

    def _get_state(self, obs_dict):
        state_parts = []
        red_agents = [a for a in self.agents if a.startswith(self.red_prefix)]
        for agent in red_agents:
            if agent in obs_dict:
                state_parts.append(obs_dict[agent].flatten())
            else:
                state_parts.append(np.zeros(self.obs_shape))
        
        if state_parts:
            state = np.concatenate(state_parts)
        else:
            state = np.zeros(self.obs_shape[0] * self.n_agents)
        
        return [state] * self.n_agents

    def _get_avail_actions(self):
        return [np.ones(self.action_space[0].n) for _ in range(self.n_agents)]

    def get_env_info(self):
        return {
            "state_shape": self.obs_shape[0] * self.n_agents,
            "obs_shape": self.obs_shape[0],
            "n_actions": self.action_space[0].n,
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
            "action_spaces": self.action_space
        }

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        self.env.close()


class AdvPursuit_w_PretrainedOpp(AdvPursuitEnv):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        pretrained_ckpt = kwargs["env_args"].get("pretrained_ckpt", None)
        self.global_reward = kwargs["env_args"].get("global_reward", False)
        
        if pretrained_ckpt:
            self.prey_policy = IDQN_AdvPursuit(
                pretrained_ckpt=pretrained_ckpt, 
                input_shape=128
            )
        else:
            self.prey_policy = None
        
        blue_agents = [a for a in self.agents if a.startswith(self.blue_prefix)]
        self.blue_obs = [np.zeros(self.obs_shape) for _ in blue_agents]

    def reset(self):
        obs, state, avail_actions = super().reset()
        
        blue_agents = [a for a in self.agents if a.startswith(self.blue_prefix)]
        self.blue_obs = [np.zeros(self.obs_shape) for _ in blue_agents]
        
        return obs, state, avail_actions

    def step(self, actions):
        red_agents = [a for a in self.agents if a.startswith(self.red_prefix)]
        blue_agents = [a for a in self.agents if a.startswith(self.blue_prefix)]
        
        action_dict = {}
        
        for i, agent in enumerate(red_agents):
            if i < len(actions):
                action_dict[agent] = int(actions[i])
        
        if self.prey_policy:
            if len(self.blue_obs) == len(blue_agents):
                blue_actions = self.prey_policy.step(
                    obss=self.blue_obs,
                    avail_actions=[np.ones(self.action_space[0].n)] * len(blue_agents)
                )
                for i, agent in enumerate(blue_agents):
                    if hasattr(self.env, 'agents') and agent not in self.env.agents:
                        continue
                    action_dict[agent] = int(blue_actions[i])
            else:
                 for agent in blue_agents:
                    if hasattr(self.env, 'agents') and agent not in self.env.agents:
                        continue
                    action_dict[agent] = self.env.action_space.sample()
        else:
            for agent in blue_agents:
                if hasattr(self.env, 'agents') and agent not in self.env.agents:
                    continue
                action_dict[agent] = self.env.action_space.sample()
        
        
        obs_dict, rewards, dones, raw_infos = self.env.step(action_dict)
        
        obs = []
        red_rewards = []
        for agent in red_agents:
            if agent in obs_dict:
                obs.append(obs_dict[agent].flatten())
                red_rewards.append(rewards.get(agent, 0.0))
            else:
                obs.append(np.zeros(self.obs_shape))
                red_rewards.append(0.0)
        
        self.blue_obs = []
        for agent in blue_agents:
            if agent in obs_dict:
                self.blue_obs.append(obs_dict[agent].flatten())
            else:
                self.blue_obs.append(np.zeros(self.obs_shape))
        
        state = self._get_state(obs_dict)
        avail_actions = self._get_avail_actions()
        
        if isinstance(dones, dict):
            done_env = dones.get("__all__", False)
        else:
            done_env = dones
            
        if self.global_reward:
            if red_rewards:
                global_rew = mean(red_rewards)
                red_rewards = [global_rew] * len(red_rewards)
            else:
                red_rewards = [0.0] * self.n_agents
        
        current_agents = self.env.agents
        red_alive = len([a for a in current_agents if a.startswith(self.red_prefix)])
        blue_alive = len([a for a in current_agents if a.startswith(self.blue_prefix)])
        
        red_win = False
        blue_win = False
        if done_env:
            if blue_alive == 0:
                red_win = True
            else:
                blue_win = True

        final_infos = []
        for i, agent in enumerate(red_agents):
            agent_info = raw_infos.get(agent, {}).copy()
            if i == 0:
                agent_info['red_team_alives'] = red_alive
                agent_info['blue_team_alives'] = blue_alive
                agent_info['red_team_win'] = red_win
                agent_info['blue_team_win'] = blue_win
            final_infos.append(agent_info)
        
        rewards = [[r] for r in red_rewards]
        dones_list = [done_env] * self.n_agents
        
        return obs, state, rewards, dones_list, final_infos, avail_actions