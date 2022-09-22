"""
Copyright 2022 Div Garg. All rights reserved.

Standalone IQ-Learn algorithm. See LICENSE for licensing terms.
"""

import torch
import numpy as np
import torch.nn.functional as F
from utils.utils import eval_mode, average_dicts, get_concat_samples, evaluate, soft_update, hard_update

# Full IQ-Learn objective with other divergences and options
def iq_loss(agent, current_Q, current_v, next_v, batch):
    args = agent.args
    gamma = agent.gamma
    obs, next_obs, action, env_reward, done, is_expert = batch

    loss_dict = {}
    # keep track of value of initial states
    v0 = agent.getV(obs[is_expert.squeeze(1), ...]).mean()
    loss_dict['v0'] = v0.item()

    #  calculate 1st term for IQ loss
    #  -E_(ρ_expert)[Q(s, a) - γV(s')]
    y = (1 - done) * gamma * next_v
    reward = (current_Q - y)[is_expert]

    with torch.no_grad():
        # Use different divergence functions (For χ2 divergence we instead add a third bellmann error-like term)
        if args.method.div == "hellinger":
            phi_grad = 1/(1+reward)**2
        elif args.method.div == "kl":
            # original dual form for kl divergence (sub optimal)
            phi_grad = torch.exp(-reward-1)
        elif args.method.div == "kl2":
            # biased dual form for kl divergence
            phi_grad = F.softmax(-reward, dim=0) * reward.shape[0]
        elif args.method.div == "kl_fix":
            # our proposed unbiased form for fixing kl divergence
            phi_grad = torch.exp(-reward)
        elif args.method.div == "js":
            # jensen–shannon
            phi_grad = torch.exp(-reward)/(2 - torch.exp(-reward))
        else:
            phi_grad = 1
    loss = -(phi_grad * reward).mean()
    loss_dict['softq_loss'] = loss.item()

    # calculate 2nd term for IQ loss, we show different sampling strategies
    if args.method.loss == "value_expert":
        # sample using only expert states (works offline)
        # E_(ρ)[Q(s,a) - γV(s')]
        value_loss = (current_v - y)[is_expert].mean()
        loss += value_loss
        loss_dict['value_loss'] = value_loss.item()

    elif args.method.loss == "value":
        # sample using expert and policy states (works online)
        # E_(ρ)[V(s) - γV(s')]
        value_loss = (current_v - y).mean()
        loss += value_loss
        loss_dict['value_loss'] = value_loss.item()

    elif args.method.loss == "v0":
        # alternate sampling using only initial states (works offline but usually suboptimal than `value_expert` startegy)
        # (1-γ)E_(ρ0)[V(s0)]
        v0_loss = (1 - gamma) * v0
        loss += v0_loss
        loss_dict['v0_loss'] = v0_loss.item()

    # alternative sampling strategies for the sake of completeness but are usually suboptimal in practice
    # elif args.method.loss == "value_policy":
    #     # sample using only policy states
    #     # E_(ρ)[V(s) - γV(s')]
    #     value_loss = (current_v - y)[~is_expert].mean()
    #     loss += value_loss
    #     loss_dict['value_policy_loss'] = value_loss.item()

    # elif args.method.loss == "value_mix":
    #     # sample by weighted combination of expert and policy states
    #     # E_(ρ)[Q(s,a) - γV(s')]
    #     w = args.method.mix_coeff
    #     value_loss = (w * (current_v - y)[is_expert] +
    #                   (1-w) * (current_v - y)[~is_expert]).mean()
    #     loss += value_loss
    #     loss_dict['value_loss'] = value_loss.item()

    else:
        raise ValueError(f'This sampling method is not implemented: {args.method.type}')

    if args.method.grad_pen:
        # add a gradient penalty to loss (Wasserstein_1 metric)
        gp_loss = agent.critic_net.grad_pen(obs[is_expert.squeeze(1), ...],
                                            action[is_expert.squeeze(1), ...],
                                            obs[~is_expert.squeeze(1), ...],
                                            action[~is_expert.squeeze(1), ...],
                                            args.method.lambda_gp)
        loss_dict['gp_loss'] = gp_loss.item()
        loss += gp_loss

    if args.method.div == "chi" or args.method.chi:  # TODO: Deprecate method.chi argument for method.div
        # Use χ2 divergence (calculate the regularization term for IQ loss using expert states) (works offline)
        y = (1 - done) * gamma * next_v

        reward = current_Q - y
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2)[is_expert].mean()
        loss += chi2_loss
        loss_dict['chi2_loss'] = chi2_loss.item()

    if args.method.regularize:
        # Use χ2 divergence (calculate the regularization term for IQ loss using expert and policy states) (works online)
        y = (1 - done) * gamma * next_v

        reward = current_Q - y
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
        loss += chi2_loss
        loss_dict['regularize_loss'] = chi2_loss.item()

    loss_dict['total_loss'] = loss.item()
    return loss, loss_dict

# Full IQ-Learn objective with other divergences and options
def ppil_loss(agent, current_r, current_Q, current_v, next_v, batch):
    args = agent.args
    gamma = agent.gamma
    obs, next_obs, action, env_reward, done, is_expert = batch
    #print(len(is_expert))
    steps=20
    for l in range(steps):
        loss_dict = {}

        current_Q = agent.critic(obs, action)
        current_r = agent.get_reward(obs, action)
        current_v = agent.getV(obs)
        if args.train.use_target:
            with torch.no_grad():
                next_v = agent.get_targetV(next_obs)
        else:
            next_v = agent.getV(next_obs)

        # keep track of value of initial states
        v0 = agent.getV(obs[is_expert.squeeze(1), ...]).mean()
        loss_dict['v0'] = v0.item()
        #Weights
        y = (1 - done) * gamma * next_v

        with torch.no_grad():
            if args.agent.online:
                delta_loss = (-current_Q  + current_r + y)[~is_expert]
                weights = torch.softmax(args.agent.tau*delta_loss, dim=0) #tau=1e1
            else:
                delta_loss_expert = (-current_Q  + current_r + y)[is_expert]
                weights_expert = torch.softmax(args.agent.tau*delta_loss_expert, dim=0) #tau=1e-1
        #  calculate 1st term for IQ loss
        #  -E_(ρ_expert)[Q(s, a) - γV(s')]
        if not args.agent.online:
            loss = - (current_r)[is_expert].mean() + (current_r)[is_expert].dot(weights_expert) 
        else:
            loss = - (current_r)[is_expert].mean() + (current_r)[~is_expert].dot(weights)

        loss_dict['reward_loss'] = loss.item()

        # calculate 2nd term for IQ loss

        
        #delta_loss = (-current_Q  + current_r + y)[~is_expert] 
        if not args.agent.online:
            logistic_loss = (-current_Q + y)[is_expert].dot(weights_expert)  #torch.logsumexp(1e1*(delta_loss.mean() - torch.log(torch.FloatTensor([delta_loss.shape[0]]))), dim=0)/1e1
        else:
            logistic_loss = (-current_Q + y)[~is_expert].dot(weights)
        loss += logistic_loss
        loss_dict['logistic_loss'] = logistic_loss.item()

        # calculate 3rd term for IQ loss, we show different sampling strategies

        if args.method.loss == "value_expert":
            # sample using only expert states (works offline)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = (current_v - y)[is_expert].mean()
            loss += value_loss
            loss_dict['value_loss'] = value_loss.item()

        elif args.method.loss == "value":
            # sample using expert and policy states (works online)
            # E_(ρ)[V(s) - γV(s')]
            value_loss = (current_v - y).mean()
            loss += value_loss
            loss_dict['value_loss'] = value_loss.item()

        elif args.method.loss == "v0":
            # alternate sampling using only initial states (works offline but usually suboptimal than `value_expert` startegy)
            # (1-γ)E_(ρ0)[V(s0)]
            v0_loss = (1 - gamma) * v0
            loss += v0_loss
            loss_dict['v0_loss'] = v0_loss.item()


        else:
            raise ValueError(f'This sampling method is not implemented: {args.method.type}')

        if args.method.grad_pen:
            # add a gradient penalty to loss (Wasserstein_1 metric)
            gp_loss = agent.critic_net.grad_pen(obs[is_expert.squeeze(1), ...],
                                                action[is_expert.squeeze(1), ...],
                                                obs[~is_expert.squeeze(1), ...],
                                                action[~is_expert.squeeze(1), ...],
                                                args.method.lambda_gp)
            loss_dict['gp_loss'] = gp_loss.item()
            loss += gp_loss

        if args.method.div == "chi" or args.method.chi:  # TODO: Deprecate method.chi argument for method.div
            # Use χ2 divergence (calculate the regularization term for IQ loss using expert states) (works offline)
            
            chi2_loss = 1/(4 * args.method.alpha) * (current_r**2)[is_expert].mean()
            loss += chi2_loss
            loss_dict['chi2_loss'] = chi2_loss.item()

        if args.method.regularize:
            # Use χ2 divergence (calculate the regularization term for IQ loss using expert and policy states) (works online)
            
            chi2_loss = 1/(4 * args.method.alpha) * (current_r**2).mean() + 1/(4*args.method.alpha)*(current_Q**2).mean()
            loss += chi2_loss
            loss_dict['regularize_loss'] = chi2_loss.item()

        loss_dict['total_loss'] = loss.item()
        #agent.critic_optimizer.param_groups[0]['lr'] = args.agent.critic_lr/np.sqrt(l+1)
        #agent.reward_optimizer.param_groups[0]['lr'] = args.agent.critic_lr/np.sqrt(l+1)
        if l < steps-1:
            agent.critic_optimizer.zero_grad()
            agent.reward_optimizer.zero_grad()
            loss.backward()
            agent.critic_optimizer.step()
            agent.reward_optimizer.step()
        if l % agent.critic_target_update_frequency == 0 and l < steps-1:
            if args.train.soft_update:
                soft_update(agent.critic_net, agent.critic_target_net,
                        agent.critic_tau)
            else:
                hard_update(agent.critic_net, agent.critic_target_net)
    return loss, loss_dict

def alternate_ppil_loss(agent, current_r, current_Q, current_v, next_v, batch, w_update=False):
    args = agent.args
    gamma = agent.gamma
    obs, next_obs, action, env_reward, done, is_expert = batch
    #print(len(is_expert))
    steps=20
    for l in range(steps):
        loss_dict = {}

        current_Q = agent.critic(obs, action)
        current_r = agent.get_reward(obs, action)
        current_v = agent.getV(obs)
        if args.train.use_target:
            with torch.no_grad():
                next_v = agent.get_targetV(next_obs)
        else:
            next_v = agent.getV(next_obs)

        # keep track of value of initial states
        v0 = agent.getV(obs[is_expert.squeeze(1), ...]).mean()
        loss_dict['v0'] = v0.item()
        #Weights
        y = (1 - done) * gamma * next_v

        with torch.no_grad():
            if args.agent.online:
                delta_loss = (-current_Q  + current_r + y)[~is_expert]
                weights = torch.softmax(args.agent.tau*delta_loss, dim=0) #tau=1e1
            else:
                delta_loss_expert = (-current_Q  + current_r + y)[is_expert]
                weights_expert = torch.softmax(args.agent.tau*delta_loss_expert, dim=0) #tau=1e-1
        #  calculate 1st term for IQ loss
        #  -E_(ρ_expert)[Q(s, a) - γV(s')]
        if not args.agent.online:
            loss = - (current_r)[is_expert].mean() + (current_r)[is_expert].dot(weights_expert) 
        else:
            loss = - (current_r)[is_expert].mean() + (current_r)[~is_expert].dot(weights)

        loss_dict['reward_loss'] = loss.item()

        # calculate 2nd term for IQ loss

        
        #delta_loss = (-current_Q  + current_r + y)[~is_expert] 
        if not args.agent.online:
            logistic_loss = (-current_Q + y)[is_expert].dot(weights_expert)  #torch.logsumexp(1e1*(delta_loss.mean() - torch.log(torch.FloatTensor([delta_loss.shape[0]]))), dim=0)/1e1
        else:
            logistic_loss = (-current_Q + y)[~is_expert].dot(weights)
        loss += logistic_loss
        loss_dict['logistic_loss'] = logistic_loss.item()

        # calculate 3rd term for IQ loss, we show different sampling strategies

        if args.method.loss == "value_expert":
            # sample using only expert states (works offline)
            # E_(ρ)[Q(s,a) - γV(s')]
            value_loss = (current_v - y)[is_expert].mean()
            loss += value_loss
            loss_dict['value_loss'] = value_loss.item()

        elif args.method.loss == "value":
            # sample using expert and policy states (works online)
            # E_(ρ)[V(s) - γV(s')]
            value_loss = (current_v - y).mean()
            loss += value_loss
            loss_dict['value_loss'] = value_loss.item()

        elif args.method.loss == "v0":
            # alternate sampling using only initial states (works offline but usually suboptimal than `value_expert` startegy)
            # (1-γ)E_(ρ0)[V(s0)]
            v0_loss = (1 - gamma) * v0
            loss += v0_loss
            loss_dict['v0_loss'] = v0_loss.item()


        else:
            raise ValueError(f'This sampling method is not implemented: {args.method.type}')

        if args.method.grad_pen:
            # add a gradient penalty to loss (Wasserstein_1 metric)
            gp_loss = agent.critic_net.grad_pen(obs[is_expert.squeeze(1), ...],
                                                action[is_expert.squeeze(1), ...],
                                                obs[~is_expert.squeeze(1), ...],
                                                action[~is_expert.squeeze(1), ...],
                                                args.method.lambda_gp)
            loss_dict['gp_loss'] = gp_loss.item()
            loss += gp_loss

        if args.method.div == "chi" or args.method.chi:  # TODO: Deprecate method.chi argument for method.div
            # Use χ2 divergence (calculate the regularization term for IQ loss using expert states) (works offline)
            
            chi2_loss = 1/(4 * args.method.alpha) * (current_r**2)[is_expert].mean()
            loss += chi2_loss
            loss_dict['chi2_loss'] = chi2_loss.item()

        if args.method.regularize:
            # Use χ2 divergence (calculate the regularization term for IQ loss using expert and policy states) (works online)
            
            chi2_loss = 1/(4 * args.method.alpha) * (current_r**2).mean() + 1/(4*args.method.alpha)*(current_Q**2).mean()
            loss += chi2_loss
            loss_dict['regularize_loss'] = chi2_loss.item()

        loss_dict['total_loss'] = loss.item()
        #agent.critic_optimizer.param_groups[0]['lr'] = args.agent.critic_lr/np.sqrt(l+1)
        #agent.reward_optimizer.param_groups[0]['lr'] = args.agent.critic_lr/np.sqrt(l+1)
        if l < steps-1:
            if w_update:
                agent.reward_optimizer.zero_grad()
                loss.backward()
                agent.reward_optimizer.step()
            else:
                agent.critic_optimizer.zero_grad()
                loss.backward()
                agent.critic_optimizer.step()
        if l % agent.critic_target_update_frequency == 0 and l < steps-1:
            if args.train.soft_update:
                soft_update(agent.critic_net, agent.critic_target_net,
                        agent.critic_tau)
            else:
                hard_update(agent.critic_net, agent.critic_target_net)
    return loss, loss_dict