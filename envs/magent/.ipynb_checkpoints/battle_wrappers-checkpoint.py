import math
from statistics import mean
import supersuit as ss
import numpy as np
import torch as th
from gym.spaces import Dict as GymDict, Box
import pettingzoo.magent as magent

from ...pretrained.magent import IDQN_Battle
from ..multiagentenv import MultiAgentEnv
from .magent_env import PettingZooEnv


MAPSIZE2N = {
    25: 20, 35: 42, 40: 64, 45: 81, 50: 100, 
    55: 121, 60: 144, 70: 196, 80: 256, 90: 324, 100: 400
}

class BattleEnv(MultiAgentEnv):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.map_name = kwargs["env_args"].get("map_name", "battle_v3")
        self.max_cycles = kwargs["env_args"].get("max_cycles", 200)
        self.seed = kwargs["env_args"].get("seed", None)
        self.map_size = kwargs["env_args"].get("map_size", 50)
        
        env = magent.battle_v3.parallel_env(
            map_size=self.map_size,
            max_cycles=self.max_cycles,
            minimap_mode=kwargs["env_args"].get("minimap_mode", True)
        )
        
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
        self.env = PettingZooEnv(env)
        
        self.agents = self.env.agents
        self.n_agents = MAPSIZE2N.get(self.map_size, 100)
        self.episode_limit = self.max_cycles
        
        obs_sample = self.env.reset()
        first_obs = obs_sample[self.agents[0]]
        self.obs_shape = (np.prod(first_obs.shape),)
        
        self.observation_space = [Box(low=-np.inf, high=np.inf, shape=self.obs_shape) 
                                for _ in range(self.n_agents)]
        self.share_observation_space = self.observation_space.copy()
        self.action_space = [gym.spaces.Discrete(self.env.action_space.n) 
                           for _ in range(self.n_agents)]

    def reset(self):
        obs_dict = self.env.reset()
        obs = []
        red_agents = [a for a in self.agents if a.startswith('red')]
        
        for agent in red_agents:
            obs.append(obs_dict[agent].flatten())
        
        state = self._get_state(obs_dict)
        avail_actions = self._get_avail_actions()
        
        return obs, state, avail_actions

    def step(self, actions):

        action_dict = {}
        red_agents = [a for a in self.agents if a.startswith('red')]
        blue_agents = [a for a in self.agents if a.startswith('blue')]
        
        for i, agent in enumerate(red_agents):
            action_dict[agent] = int(actions[i])
        

        for agent in blue_agents:
            action_dict[agent] = self.env.action_space.sample()
        

        obs_dict, rewards, dones, infos = self.env.step(action_dict)
        

        obs = []
        red_rewards = []
        red_alive = 0
        blue_alive = 0
        
        for agent in red_agents:
            if agent in obs_dict:
                obs.append(obs_dict[agent].flatten())
                red_rewards.append(rewards[agent])
                red_alive += 1
            else:
                obs.append(np.zeros(self.obs_shape))
                red_rewards.append(0.0)
        
        for agent in blue_agents:
            if agent in obs_dict:
                blue_alive += 1
        
        state = self._get_state(obs_dict)
        avail_actions = self._get_avail_actions()
        done = dones["__all__"]
        

        infos = [{"red_alive": red_alive, "blue_alive": blue_alive} for _ in range(self.n_agents)]
        
        rewards = [[r] for r in red_rewards]
        dones = [done] * self.n_agents
        
        return obs, state, rewards, dones, infos, avail_actions

    def _get_state(self, obs_dict):

        state_parts = []
        red_agents = [a for a in self.agents if a.startswith('red')]
        for agent in red_agents:
            if agent in obs_dict:
                state_parts.append(obs_dict[agent].flatten())
        
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


class Battle_w_PretrainedOpp(BattleEnv):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        pretrained_ckpt = kwargs["env_args"].get("pretrained_ckpt", None)
        self.global_reward = kwargs["env_args"].get("global_reward", False)
        
        if pretrained_ckpt:
            self.blue_policy = IDQN_Battle(
                pretrained_ckpt=pretrained_ckpt,
                input_shape=98
            )
        else:
            self.blue_policy = None
        
        self.blue_obs = None

    def reset(self):
        obs, state, avail_actions = super().reset()
        
        blue_agents = [a for a in self.agents if a.startswith('blue')]
        obs_dict = self.env.reset()
        self.blue_obs = [obs_dict[agent].flatten() for agent in blue_agents]
        
        return obs, state, avail_actions

    def step(self, actions):
        red_agents = [a for a in self.agents if a.startswith('red')]
        blue_agents = [a for a in self.agents if a.startswith('blue')]
        
        action_dict = {}
        
        for i, agent in enumerate(red_agents):
            action_dict[agent] = int(actions[i])
        
        if self.blue_policy and self.blue_obs:
            blue_actions = self.blue_policy.step(
                obss=self.blue_obs,
                avail_actions=[np.ones(self.action_space[0].n)] * len(blue_agents)
            )
            for i, agent in enumerate(blue_agents):
                action_dict[agent] = int(blue_actions[i])
        else:
            for agent in blue_agents:
                action_dict[agent] = self.env.action_space.sample()
        
        obs_dict, rewards, dones, infos = self.env.step(action_dict)
        
        obs = []
        red_rewards = []
        red_alive = 0
        blue_alive = 0
        
        for agent in red_agents:
            if agent in obs_dict:
                obs.append(obs_dict[agent].flatten())
                red_rewards.append(rewards[agent])
                red_alive += 1
            else:
                obs.append(np.zeros(self.obs_shape))
                red_rewards.append(0.0)
        
        for agent in blue_agents:
            if agent in obs_dict:
                blue_alive += 1
        
        self.blue_obs = []
        for agent in blue_agents:
            if agent in obs_dict:
                self.blue_obs.append(obs_dict[agent].flatten())
            else:
                self.blue_obs.append(np.zeros(self.obs_shape))
        
        state = self._get_state(obs_dict)
        avail_actions = self._get_avail_actions()
        done = dones["__all__"]
        
        if self.global_reward:
            global_reward = mean(red_rewards)
            red_rewards = [global_reward] * len(red_rewards)
        
        infos = [{"red_alive": red_alive, "blue_alive": blue_alive} for _ in range(self.n_agents)]
        
        rewards = [[r] for r in red_rewards]
        dones = [done] * self.n_agents
        
        return obs, state, rewards, dones, infos, avail_actions