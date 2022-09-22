import torch
import gym
import random
import argparse
import numpy as np
import sys
import pickle
sys.path.insert(0,"../")
from lpimi.envs.wrappers import TimeLimit

parser = argparse.ArgumentParser(description='IM Learning')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-traj-path', metavar='G',
                    help='path of the expert trajectories')
parser.add_argument('--optimizer', metavar='G', default="forb",
                    help='optimizer')
parser.add_argument('--linear', action='store_true', default=False,
                    help='Use linear model for policy and reward')
parser.add_argument('--gamma', type=float, default=0.9, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--learning-rate-theta', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--n-trajs', type=int, default=100, metavar='N',
                    help='minimal batch size for evaluation (default: 2048)')
parser.add_argument('--K', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
args = parser.parse_args()

def env_creator(name, seed,time_limit):
    from rllib.environment import GymEnvironment
    import lpimi.envs.scuric
    cfg={} # TODO: set configs according to name
    return TimeLimit(GymEnvironment(name, seed=seed, **cfg), time_limit)

expert_dataset = torch.load(open("expert_traj/"+args.expert_traj_path, "rb"))

class CartPoleFeat(torch.nn.Module):
    def __init__(self,input_dim,feat_size,hidden_dim=20):
        super(CartPoleFeat, self).__init__()
        self.frozen=torch.nn.Sequential(*[
            torch.nn.Linear(input_dim,hidden_dim),
           torch.nn.Tanh()
        ])
        self.linear=torch.nn.Sequential(*[torch.nn.Linear(hidden_dim,feat_size)])
    def forward(self,inp,done=False)->torch.Tensor:
        self.frozen.eval() # enforce frozen
        return self.linear(self.frozen(inp))
class OneHot(torch.nn.Module):
    def __init__(self, input_dim):
        super(OneHot, self).__init__()
        self.input_dim=input_dim
    def forward(self, inp):
        index = inp[0]*env.action_space.n + inp[1]
        out = torch.zeros(self.input_dim)
        out[index]=1
        return out

gamma = args.gamma
env = env_creator(args.env_name,seed=0,time_limit=200)
reset_env = env_creator(args.env_name,seed=0,time_limit=200)
phi = OneHot(input_dim=env.action_space.n*env.observation_space.n)

class Policy:
    def __init__(self, theta, phi, alpha):
        self.phi = phi
        self.alpha = alpha
        self.theta_sum = theta
        self.old_increment = 0
        
    def get_probabilities(self, s):
        deltas = torch.tensor([self.alpha*(self.phi(torch.tensor([s,a])).dot(self.theta_sum)) for a in range(env.action_space.n)])
        exp_deltas = deltas
        return torch.softmax(exp_deltas, axis=0) #exp_deltas/torch.sum(exp_deltas)
    
    def update(self, increment):
        self.theta_sum = increment


expert_state_action_pairs = []
m = len(expert_dataset["trajectories"])
for traj in expert_dataset["trajectories"]:
    for t,pair in enumerate(traj):
        state, action, next_state,_,_ = pair
        expert_state_action_pairs.append([torch.where(state), torch.where(action), torch.where(next_state)])

def sample_batch(policy, N, counter):
    batch = []
    done=False
    for n in range(N):
        traj = []
        s = env.reset()
        while not done:
            probs = policy.get_probabilities(torch.tensor(s))
            distr = torch.distributions.Categorical(probs)
            a = distr.sample()
            #a = np.random.choice(env.action_space.n, p=probs)
            s_new,_,done,_ = env.step(a)
            counter += 1
            traj.append((s,a,s_new))
            s = s_new
        batch.append(traj)
        done=False
    return batch, counter

def estimate_return(policy,N):
    done=False
    returns = []
    for n in range(N):
        traj = []
        tot_reward = 0
        s = env.reset()
        while not done:
            probs = policy.get_probabilities(torch.tensor(s))
            distr = torch.distributions.Categorical(probs)
            a = distr.sample()
            #a = np.random.choice(env.action_space.n, p=probs)
            s_new,r,done,_ = env.step(a)
            traj.append((s,a,s_new))
            tot_reward += r
            s = s_new
        done=False
        returns.append(tot_reward)
    return np.mean(returns), np.std(returns)

class AdamOptimizer:
    def __init__(self, size, lr=0.5, b1=.9, b2=.99, epsilon=10e-8):
        self.type = "adam"

        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.epsilon = epsilon

        self.step = 0

        self.m = torch.zeros(size)
        self.v = torch.zeros(size)

    def update(self, grad):
        self.step += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        m_hat = self.m / (1 - self.b1 ** self.step)
        v_hat = self.v / (1 - self.b2 ** self.step)

        return (self.lr * m_hat) / (torch.sqrt(v_hat) + self.epsilon)

def run_IQ_Learn(optimizer, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    theta = torch.zeros(env.action_space.n*env.observation_space.n)
    policy = Policy(theta, phi, 1)
    if optimizer=="adam":
        opt_theta = AdamOptimizer(env.action_space.n*env.observation_space.n, lr=args.learning_rate_theta)
    
    returns = []
    steps = []
    
    counter = 0
    G = estimate_return(policy, 10)
    returns.append(G[0])
    steps.append(counter)
    
    for _ in range(args.K):
        n_trajs=args.n_trajs
        batch = random.sample(expert_state_action_pairs, n_trajs)        
        
        old_gradient = 0
        #w = torch.zeros(env.action_space.n*env.observation_space.n)
        #theta = torch.zeros(env.action_space.n*env.observation_space.n)
        gradient=0
        for b in batch:
            state, action, next_state = b
            feature_s_a = policy.phi(torch.tensor([state,action]))
            Q_s_a = feature_s_a.squeeze(0).dot(theta)
            features_next_state = torch.stack([policy.phi(torch.tensor([next_state,[a]])) 
                                               for a in range(env.action_space.n)])
            Q_next_state = features_next_state.matmul(theta)
            V_next_state = torch.logsumexp(Q_next_state, axis=0)
            
            probs = torch.softmax(Q_next_state, axis=0)
            gradient += (feature_s_a - gamma*features_next_state.T.matmul(probs))*(1 - 0.5*Q_s_a + 0.5*gamma*V_next_state)
        gradient = gradient/n_trajs
        
        batch, counter = sample_batch(policy, n_trajs, counter)
        
        batch_flatten = []
        for traj in batch:
            for _,pair in enumerate(traj):
                batch_flatten.append(pair)
        gradient_2 = 0
        for b in batch_flatten:
            state, action, next_state = b
            features_state = torch.stack([policy.phi(torch.tensor([[state],[a]])) 
                                               for a in range(env.action_space.n)])
            features_next_state = torch.stack([policy.phi(torch.tensor([[next_state],[a]])) 
                                               for a in range(env.action_space.n)])
            Q_next_state = features_next_state.matmul(theta)
            Q_state = features_state.matmul(theta)
            
            probs_next_state = torch.softmax(Q_next_state, axis=0)
            probs_state = torch.softmax(Q_state, axis=0)
            
            gradient_2 = features_state.T.matmul(probs_state) - gamma*features_next_state.T.matmul(probs_next_state)
        gradient = gradient - gradient_2/n_trajs
        if optimizer=="forb":
            theta += args.learning_rate_theta*(2*gradient - old_gradient)
            old_gradient = gradient.clone()
        elif optimizer=="sgd":
            theta += args.learning_rate_theta*gradient
        elif optimizer=="adam":
            theta += opt_theta.update(gradient.squeeze(0))
        policy.update(theta)
        
        G = estimate_return(policy, 10)
        returns.append(G[0])
        steps.append(counter)
        
        print(counter, G)
        with open("results/iq_learn/"+args.env_name+str(seed)
                +args.optimizer+"n_trajs"+str(args.n_trajs)
                +"_lr_theta"+str(args.learning_rate_theta)+".pt","wb") as f:
            pickle.dump((returns, steps), f)
    return policy

print("seed", args.seed)
print("_______________________________________________")

policy = run_IQ_Learn(optimizer=args.optimizer, seed = args.seed)

