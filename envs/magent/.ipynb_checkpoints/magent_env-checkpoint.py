import gym
from gym.spaces import Box, Dict as GymDict
import numpy as np
from ..multiagentenv import MultiAgentEnv

class PettingZooEnv(gym.Env):
    def __init__(self, env):
        self.par_env = env
        self.agents = self.par_env.possible_agents
        self.observation_spaces = self.par_env.observation_spaces
        self.action_spaces = self.par_env.action_spaces
        self.observation_space = self.observation_spaces[self.agents[0]]
        self.action_space = self.action_spaces[self.agents[0]]

    def reset(self):
        return self.par_env.reset()

    def step(self, action_dict):
        obss, rews, dones, infos = self.par_env.step(action_dict)
        dones["__all__"] = all(dones.values())
        return obss, rews, dones, infos

    def close(self):
        self.par_env.close()

    def seed(self, seed=None):
        self.par_env.seed(seed)

    def render(self, mode="human"):
        return self.par_env.render(mode)