import datetime
import os
import random
import time
from collections import deque
from itertools import count
import types
import pickle

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from torch.autograd import Variable, grad

from wrappers.atari_wrapper import LazyFrames
from logger import Logger
from make_envs import make_env
from memory import Memory
from agent import make_agent
from utils import eval_mode, get_concat_samples, evaluate, soft_update, hard_update

expert_memory_replay = Memory(10000//2, 0)
expert_memory_replay.load(hydra.utils.to_absolute_path(f'experts/Acrobot-v1_1000.pkl'),
                            num_trajs=100,
                            sample_freq=5,
                            seed=42)
print(expert_memory_replay.sample(batch_size=40))
print(dir(expert_memory_replay))