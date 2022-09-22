"""
Copyright 2022 Div Garg. All rights reserved.

Example training code for IQ-Learn which minimially modifies `train_rl.py`.
"""

import datetime
import os
import random
import time
from collections import deque
from itertools import count
import types

import hydra
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter

from wrappers.atari_wrapper import LazyFrames
from make_envs import make_env
from dataset.memory import Memory
from agent import make_agent
from utils.utils import eval_mode, average_dicts, get_concat_samples, evaluate, soft_update, hard_update
from utils.logger import Logger
from iq import iq_loss, ppil_loss, alternate_ppil_loss

torch.set_num_threads(2)

def save(agent, epoch, args, output_dir='results/bc/'):
    if epoch % args.save_interval == 0:
        if args.method.type == "sqil":
            name = f'sqil_{args.env.name}'
        else:
            name = f'iq_{args.env.name}'

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        agent.save(f'{output_dir}/{args.agent.name}_{name}')

def get_args(cfg: DictConfig):
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.hydra_base_dir = os.getcwd()
    print(OmegaConf.to_yaml(cfg))
    return cfg


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    args = get_args(cfg)
    wandb.init(project=args.project_name, entity='viano',
               sync_tensorboard=True, reinit=True, config=args)

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == 'cuda' and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, args.env.name, args.exp_name, args.method.type, str(args.seed), ts_str)
    writer = SummaryWriter(log_dir=log_dir)
    env_args = args.env
    env = make_env(args)
    eval_env = make_env(args)

    # Seed envs
    env.seed(args.seed)
    eval_env.seed(args.seed + 10)

    REPLAY_MEMORY = int(env_args.replay_mem)
    INITIAL_MEMORY = int(env_args.initial_mem)
    EPISODE_STEPS = int(env_args.eps_steps)
    EPISODE_WINDOW = int(env_args.eps_window)
    LEARN_STEPS = int(env_args.learn_steps)
    INITIAL_STATES = 128  # Num initial states to use to calculate value of initial state distribution s_0

    agent_bc = make_agent(env, args)

    if args.pretrain:
        pretrain_path = hydra.utils.to_absolute_path(args.pretrain)
        if os.path.isfile(pretrain_path):
            print("=> loading pretrain '{}'".format(args.pretrain))
            agent_bc.load(pretrain_path)
        else:
            print("[Attention]: Did not find checkpoint {}".format(args.pretrain))

    # Load expert data
    expert_memory_replay = Memory(REPLAY_MEMORY//2, args.seed)
    expert_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.env.demo}'),
                              num_trajs=args.expert.demos,
                              sample_freq=args.expert.subsample_freq,
                              seed=args.seed + 42)
    print(f'--> Expert memory size: {expert_memory_replay.size()}')
    expert_batch = expert_memory_replay.get_samples(REPLAY_MEMORY//2, device)

    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch
    epochs =1000
    criterion = nn.CrossEntropyLoss()
    net = BCNet(env.observation_space.n,env.action_space.n, 64, args)
    for epoch in range(epochs):
        predicted_actions,_ = net(expert_obs)
        loss = criterion(predicted_actions, expert_action)
        agent_bc.actor_optimizer.zero_grad()
        loss.backward()
        agent_bc.actor_optimizer.step()
        print(loss.item())
        save(agent_bc, epoch, args)
                    
    eval_returns, eval_timesteps = evaluate(agent_bc, eval_env, num_episodes=args.eval.eps)
    returns = np.mean(eval_returns)
    returns_std = np.std(eval_returns)
    writer.add_scalar('Rewards/eval_rewards', returns,  
                                  global_step=1)
    writer.add_scalar('Rewards/std_rewards', returns_std,  
                                  global_step=1)
if __name__ == "__main__":
    main()

class BCNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim,args):
        super(BCNet, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.args = args

        self.model= torch.nn.Sequential(nn.Linear(obs_dim,hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim,hidden_dim), 
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim,action_dim),
                                        ) 

    def forward(self, obs):

        x = self.model(obs)

        return x