from functools import partial
import sys
import os

from .magent.adv_pursuit_wrappers import AdvPursuit_w_PretrainedOpp
from .magent.battle_wrappers import Battle_w_PretrainedOpp

def env_fn(env, **kwargs):
    return env(**kwargs)

REGISTRY = {}

REGISTRY["MAgent_AdvPursuit"] = partial(env_fn, env=AdvPursuit_w_PretrainedOpp)
REGISTRY["MAgent_Battle"] = partial(env_fn, env=Battle_w_PretrainedOpp)

import socket
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['train_sc.py'])


