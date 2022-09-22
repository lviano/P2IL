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

torch.set_num_threads(2)


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

    env_args = args.env 

    env = make_env(args)
    eval_env = make_env(args)
    # Seed envs
    env.seed(args.seed)
    eval_env.seed(args.seed + 10)

    REPLAY_MEMORY = int(env_args.replay_mem)
    INITIAL_MEMORY = int(env_args.initial_mem)
    UPDATE_STEPS = int(env_args.update_steps)
    EPISODE_STEPS = int(env_args.eps_steps)
    EPISODE_WINDOW = int(env_args.eps_window)
    LEARN_STEPS = int(env_args.learn_steps)

    INITIAL_STATES = 128

    agent = make_agent(env, args)

    if args.pretrain:
        pretrain_path = hydra.utils.to_absolute_path(args.pretrain)
        if os.path.isfile(pretrain_path):
            print("=> loading pretrain '{}'".format(args.pretrain))
            agent.load(pretrain_path)
        else:
            print("[Attention]: Do not find checkpoint {}".format(args.pretrain))

    expert_memory_replay = Memory(REPLAY_MEMORY//2, args.seed)
    expert_memory_replay.load(hydra.utils.to_absolute_path(f'experts/{args.env.demo}'),
                              num_trajs=args.eval.demos,
                              sample_freq=args.eval.subsample_freq,
                              seed=args.seed + 42)
    print(f'--> Expert memory size: {expert_memory_replay.size()}')

    online_memory_replay = Memory(REPLAY_MEMORY//2, args.seed+1)

    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args.log_dir, args.env.name, args.exp_name, args.method.type, str(args.seed), ts_str)
    writer = SummaryWriter(log_dir=log_dir)
    print(f'--> Saving logs at: {log_dir}')
    # TODO: Fix logging
    logger = Logger(args.log_dir)

    steps = 0

    # track avg. reward and scores
    scores_window = deque(maxlen=EPISODE_WINDOW)  # last N scores
    rewards_window = deque(maxlen=EPISODE_WINDOW)  # last N rewards
    best_eval_returns = -np.inf

    learn_steps = 0
    begin_learn = False
    episode_reward = 0
    eval_rewards = []
    rewards = []
    ss = []
    state_0 = [env.reset()] * INITIAL_STATES
    if isinstance(state_0[0], LazyFrames):
        state_0 = np.array(state_0) / 255.0
    state_0 = torch.FloatTensor(state_0).to(args.device)
    print(state_0.shape)

    for epoch in count():
        state = env.reset()
        episode_reward = 0
        done = False
        for episode_step in range(EPISODE_STEPS):
            if steps < args.num_seed_steps:
                action = env.action_space.sample()  # Sample random action
            else:
                with eval_mode(agent):
                    action = agent.choose_action(state, sample=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1

            if learn_steps % args.env.eval_interval == 0:
                eval_returns, eval_timesteps = evaluate(agent, eval_env, num_episodes=args.eval.eps)
                returns = np.mean(eval_returns)
                learn_steps += 1  # To prevent repeated eval at timestep 0
                writer.add_scalar('Rewards/eval_rewards', returns,
                                  global_step=learn_steps)
                eval_rewards.append(returns)
                print('EVAL\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, returns))
                writer.add_scalar(
                    'Success/eval', np.mean((np.array(eval_returns) > 200)), global_step=epoch)

                if returns > best_eval_returns:
                    best_eval_returns = returns
                    wandb.run.summary["best_returns"] = best_eval_returns
                    save(agent, epoch, args, output_dir='results_best')

            # allow infinite bootstrap
            done_no_lim = done
            if str(env.__class__.__name__).find('TimeLimit') >= 0 and episode_step + 1 == env._max_episode_steps:
                done_no_lim = 0
            online_memory_replay.add((state, next_state, action, reward, done_no_lim))

            if online_memory_replay.size() > INITIAL_MEMORY:
                if begin_learn is False:
                    print('learn begin!')
                    begin_learn = True
                
                learn_steps += 1
                if learn_steps == LEARN_STEPS:
                    print('Finished!')
                    wandb.finish()
                    return

                ######
                # IRL Modification
                agent.irl_update = types.MethodType(irl_update, agent)
                agent.ilr_update_critic = types.MethodType(ilr_update_critic, agent)
                losses = agent.irl_update(online_memory_replay,
                                          expert_memory_replay, logger, learn_steps)
                ######

                if learn_steps % args.log_interval == 0:
                    for key, loss in losses.items():
                        writer.add_scalar(key, loss, global_step=learn_steps)

            if done:
                break
            state = next_state

        writer.add_scalar('episodes', epoch, global_step=learn_steps)

        rewards_window.append(episode_reward)
        scores_window.append(float(episode_reward > 200))
        writer.add_scalar('Rewards/train_reward', np.mean(rewards_window), global_step=epoch)
        writer.add_scalar('Success/train', np.mean(scores_window), global_step=epoch)

        print('TRAIN\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, np.mean(rewards_window)))
        save(agent, epoch, args, output_dir='results')
        rewards.append(np.mean(rewards_window))
        ss.append(steps)

        if args.env.name in ["Hopper-v2", "HalfCheetah-v2"]:
            with open("../../../pickle_results/"+args.method.type+"/"+args.env.name+str(args.seed)
                        +"n_trajs"+str(env_args.replay_mem)
                        +"_lr_w"+str(args.agent.critic_lr)
                        +"_lr_theta"+str(args.agent.critic_lr)
                        +"_lr_actor"+str(args.agent.actor_lr)+".pt","wb") as f:
                print("Saving Pickle")
                pickle.dump((rewards, eval_rewards, ss), f)
        else:
            if args.method.type=="logistic_offline":
                with open("../../../pickle_results/"+args.method.type+"/"+args.env.name+str(args.seed)
                        +"n_trajs"+str(args.eval.demos)
                        +"_lr_w"+str(args.agent.critic_lr)
                        +"_lr_theta"+str(args.agent.critic_lr)+".pt","wb") as f:
                    print("Saving Pickle")
                    pickle.dump((rewards, eval_rewards, ss), f)
            elif args.method.type=="iq":
                with open("../../../pickle_results/"+args.method.type+"_offline/"+args.env.name+str(args.seed)
                        +"n_trajs"+str(args.eval.demos)+".pt","wb") as f:
                    print("Saving Pickle")
                    pickle.dump((rewards, eval_rewards, ss), f)
            else:
                with open("../../../pickle_results/"+args.method.type+"/"+args.env.name+str(args.seed)
                        +"n_trajs"+str(env_args.replay_mem)
                        +"_lr_w"+str(args.agent.critic_lr)
                        +"_lr_theta"+str(args.agent.critic_lr)+".pt","wb") as f:
                    print("Saving Pickle")
                    pickle.dump((rewards, eval_rewards, ss), f)

def save(agent, epoch, args, output_dir='results'):
    if epoch % args.save_interval == 0:
        if args.method.type == "sqil":
            name = f'sqil_{args.env.name}'
        else:
            name = f'iq_{args.env.name}'

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        agent.save(f'{output_dir}/{args.agent.name}_{name}')


# Minimal IQ-Learn objective
def iq_learn_update(self, policy_batch, expert_batch, logger, step):
    args = self.args
    policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch

    if args.only_expert_states:
        expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done

    obs, next_obs, action, reward, done, is_expert = get_concat_samples(
        policy_batch, expert_batch, args)

    if self.actor:
        policy_next_actions, policy_log_prob, _ = self.actor.sample(policy_next_obs)

    losses = {}

    ######
    # IQ-Learn minimal implementation with X^2 divergence (~15 lines)
    # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
    current_Q = self.critic(obs, action)
    y = (1 - done) * self.gamma * self.getV(next_obs)
    if args.train.use_target:
        with torch.no_grad():
            y = (1 - done) * self.gamma * self.get_targetV(next_obs)

    reward = (current_Q - y)[is_expert]
    loss = -(reward).mean()

    # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
    value_loss = (self.getV(obs) - y).mean()
    loss += value_loss

    # Use χ2 divergence (adds a extra term to the loss)
    chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
    loss += chi2_loss
    ######

    self.critic_optimizer.zero_grad()
    loss.backward()
    self.critic_optimizer.step()
    return loss


# Full IQ-Learn objective with other divergences and options
def ilr_update_critic(self, policy_batch, expert_batch, logger, step):
    args = self.args
    policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch

    if args.only_expert_states:
        # Use policy actions instead of experts actions for IL with only observations
        expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done

    obs, next_obs, action, reward, done, is_expert = get_concat_samples(
        policy_batch, expert_batch, args)

    losses = {}
    # keep track of v0
    v0 = self.getV(expert_obs).mean()
    losses['v0'] = v0.item()

    if args.method.type == "sqil":
        with torch.no_grad():
            target_Q = reward + (1 - done) * self.gamma * self.get_targetV(next_obs)

        current_Q = self.critic(obs, action)
        bell_error = F.mse_loss(current_Q, target_Q, reduction='none')
        loss = (bell_error[is_expert]).mean() + \
            args.method.sqil_lmbda * (bell_error[~is_expert]).mean()
        losses['sqil_loss'] = loss.item()

    elif args.method.type == "iq":
        # our method, calculate 1st term of loss
        #  -E_(ρ_expert)[Q(s, a) - γV(s')]
        current_Q = self.critic(obs, action)
        next_v = self.getV(next_obs)
        y = (1 - done) * self.gamma * next_v

        if args.train.use_target:
            with torch.no_grad():
                next_v = self.get_targetV(next_obs)
                y = (1 - done) * self.gamma * next_v

        reward = (current_Q - y)[is_expert]

        with torch.no_grad():
            if args.method.div == "hellinger":
                phi_grad = 1/(1+reward)**2
            elif args.method.div == "kl":
                phi_grad = torch.exp(-reward-1)
            elif args.method.div == "kl2":
                phi_grad = F.softmax(-reward, dim=0) * reward.shape[0]
            elif args.method.div == "kl_fix":
                phi_grad = torch.exp(-reward)
            elif args.method.div == "js":
                phi_grad = torch.exp(-reward)/(2 - torch.exp(-reward))
            else:
                phi_grad = 1
        loss = -(phi_grad * reward).mean()
        losses['softq_loss'] = loss.item()

        if args.method.loss == "v0":
            # calculate 2nd term for our loss
            # (1-γ)E_(ρ0)[V(s0)]
            v0_loss = (1 - self.gamma) * v0
            loss += v0_loss
            losses['v0_loss'] = v0_loss.item()

        elif args.method.loss == "value":
            # alternative 2nd term for our loss (use expert and policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = (self.getV(obs) - y).mean()
            loss += value_loss
            losses['value_loss'] = value_loss.item()

        elif args.method.loss == "value_policy":
            # alternative 2nd term for our loss (use only policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = (self.getV(obs) - y)[~is_expert].mean()
            loss += value_loss
            losses['value_policy_loss'] = value_loss.item()

        elif args.method.loss == "value_expert":
            # alternative 2nd term for our loss (use only expert states)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = (self.getV(obs) - y)[is_expert].mean()
            loss += value_loss
            losses['value_loss'] = value_loss.item()

        elif args.method.loss == "value_mix":
            # alternative 2nd term for our loss (use expert and policy states)
            # E_(ρ)[Q(s,a) - γV(s')]
            w = args.method.mix_coeff
            value_loss = (w * (self.getV(obs) - y)[is_expert] +
                          (1-w) * (self.getV(obs) - y)[~is_expert]).mean()
            loss += value_loss
            losses['value_loss'] = value_loss.item()

        elif args.method.loss == "skip":
            # No loss
            pass
    elif args.method.type == "logistic_q_only":
        if args.method.loss == "value":
            current_Q = self.critic(obs, action)
            next_v = self.getV(next_obs)
            y = (1 - done) * self.gamma * next_v
            #First Term
            loss = - (current_Q - y)[is_expert].mean()
            #Second Term
            value_loss = torch.exp(self.getV(obs) - y).mean()
            loss += torch.log(value_loss)

            #Third Term
            v0_loss = (1 - self.gamma) * v0
            loss += v0_loss
            losses['value'] = loss.item()
    elif args.method.type == "logistic":
        if args.method.loss == "value":
            current_Q = self.critic(obs, action)
            current_r = self.get_reward(obs, action)
            next_v = self.getV(next_obs)
            y = (1 - done) * self.gamma * next_v
            #First Term
            loss = - (current_r)[is_expert].mean()
            #Second Term
            value_loss = torch.logsumexp(1e-4*(-current_Q + current_r + y)[~is_expert], dim=0)/1e-4
            #value_loss = torch.logsumexp(10*(-current_Q + current_r + y), dim=0)/10
            
            loss += value_loss #[0]

            #Third Term
            #v0_loss = (1 - self.gamma) * v0
            #loss += v0_loss
            value_loss = (self.getV(obs) - y)[is_expert].mean()
            loss += value_loss
            losses['value'] = loss.item()
    elif args.method.type == "logistic_offline":
        if args.method.loss == "value":
            current_Q = self.critic(obs, action)
            current_r = self.get_reward(obs, action)
            next_v = self.getV(next_obs)
            y = (1 - done) * self.gamma * next_v
            #First Term
            loss = - (current_r)[is_expert].mean()
            #Second Term
            value_loss = torch.logsumexp(10*(-current_Q + current_r + y)[is_expert], dim=0)/10
            #value_loss = torch.logsumexp(10*(-current_Q + current_r + y), dim=0)/10
            
            loss += value_loss #[0]

            #Third Term
            #v0_loss = (1 - self.gamma) * v0
            #loss += v0_loss
            #w = args.method.mix_coeff
            #value_loss = (w * (self.getV(obs) - y)[is_expert] +
            #              (1-w) * (self.getV(obs) - y)[~is_expert]).mean()
            #loss += value_loss
            value_loss = (self.getV(obs) - y)[is_expert].mean()
            loss += value_loss
            losses['value'] = loss.item()
    elif args.method.type == "logistic_primal_dual":
        if args.method.loss == "value":
            current_Q = self.critic(obs, action)
            current_r = self.get_reward(obs, action)
            next_v = self.getV(next_obs)
            y = (1 - done) * self.gamma * next_v
            #First Term
            loss_reward = - (current_r)[is_expert].mean() + (current_r)[~is_expert].mean()
            
            self.reward_optimizer.zero_grad()
            loss_reward.backward()
            self.reward_optimizer.step()
            #Second Term
            current_r = self.get_reward(obs, action)
            value_loss = torch.logsumexp(10*(-current_Q + current_r.detach() + y)[~is_expert], dim=0)/10
            #value_loss = torch.logsumexp(10*(-current_Q + current_r + y), dim=0)/10
            
            loss = value_loss #[0]

            #Third Term
            v0_loss = (1 - self.gamma) * v0
            loss += v0_loss
            losses['value'] = loss.item()
    elif args.method.type=="logistic_unbiased":
        if args.method.loss == "value":
            current_Q = self.critic(obs, action)
            current_r = self.get_reward(obs, action)
            next_v = self.getV(next_obs)
            y = (1 - done) * self.gamma * next_v
            delta = (-current_Q + current_r + y)[~is_expert]
            with torch.no_grad():
                z = torch.softmax(1*delta, dim=0)
            
            loss = - (current_r)[is_expert].mean() 

            loss += torch.mean(z*delta)

            #Third Term
            v0_loss = (1 - self.gamma) * v0
            loss += v0_loss
            losses['value'] = loss.item()

    else:
        raise ValueError(f'This method is not implemented: {args.method.type}')

    if args.method.grad_pen:
        # add a gradient penalty to loss (W1 metric)
        gp_loss = self.critic_net.grad_pen(expert_obs, expert_action,
                                           policy_obs, policy_action, args.method.lambda_gp)
        #losses['gp_loss'] = gp_loss.item()
        loss += gp_loss

        gp_loss = self.reward_net.grad_pen(expert_obs, expert_action,
                                           policy_obs, policy_action, args.method.lambda_gp)
        #losses['gp_loss'] = gp_loss.item()
        loss += gp_loss

    if args.method.div == "chi" or args.method.chi:  # TODO: Deprecate method.chi argument for method.div
        # Use χ2 divergence (adds a extra term to the loss)
        if args.method.type=="iq":
            if args.train.use_target:
                with torch.no_grad():
                    next_v = self.get_targetV(next_obs)
            else:
                next_v = self.getV(next_obs)

            y = (1 - done) * self.gamma * next_v

            current_Q = self.critic(obs, action)
            reward = current_Q - y
            chi2_loss = 1/(4 * args.method.alpha) * (reward**2)[is_expert].mean()
            loss += chi2_loss
            losses['chi2_loss'] = chi2_loss.item()
        else:
            current_r = self.get_reward(obs, action)
            chi2_loss = 1/(4 * args.method.alpha) * (current_r**2)[is_expert].mean()
            loss += chi2_loss
    if args.method.regularize:
        if args.method.type=="iq":
            # Use χ2 divergence (adds a extra term to the loss)
            if args.train.use_target:
                with torch.no_grad():
                    next_v = self.get_targetV(next_obs)
            else:
                next_v = self.getV(next_obs)

            y = (1 - done) * self.gamma * next_v

            current_Q = self.critic(obs, action)
            reward = current_Q - y
            chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
            loss += chi2_loss
            losses['regularize_loss'] = chi2_loss.item()
        else:
            current_r = self.get_reward(obs, action)
            chi2_loss = 1/(4 * args.method.alpha) * (current_r**2).mean()
            loss += chi2_loss
    losses['total_loss'] = loss.item()
    if args.method.type == "logistic" or args.method.type == "logistic_offline" or args.method.type=="logistic_unbiased":
        # Optimize the critic and the reward
        self.critic_optimizer.zero_grad()
        self.reward_optimizer.zero_grad()
        loss.backward()
        # step critic
        self.critic_optimizer.step()
        self.reward_optimizer.step()
    else:
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        loss.backward()
        # step critic
        self.critic_optimizer.step()
    return losses


def irl_update(self, policy_buffer, expert_buffer, logger, step):
    policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

    losses = self.ilr_update_critic(policy_batch, expert_batch, logger, step)

    if self.actor and step % self.actor_update_frequency == 0:
        if not self.args.agent.vdice_actor:

            if self.args.offline:
                obs = expert_batch[0]
            else:
                # Use both policy and expert observations
                obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)

            if self.args.num_actor_updates:
                for i in range(self.args.num_actor_updates):
                    actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)

            # actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)
            losses.update(actor_alpha_losses)

    if step % self.critic_target_update_frequency == 0:
        if self.args.train.soft_update:
            soft_update(self.critic_net, self.critic_target_net,
                        self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    return losses


if __name__ == "__main__":
    main()
