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
parser.add_argument('--learning-rate-w', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--learning-rate-theta', type=float, default=3e-4, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--n-trajs', type=int, default=100, metavar='N',
                    help='minimal batch size for evaluation (default: 2048)')
parser.add_argument('--n-dual-updates', type=int, default=3, metavar='N',
                    help='number of dual updates (default: 3)')
parser.add_argument('--K', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--T', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--save-model-interval', type=int, default=0, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--primal-dual', action="store_true", default=False)
parser.add_argument('--incremental', action="store_true", default=False)
parser.add_argument('--biased-sgd', action="store_true", default=False)
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

def compute_expert_fev(expert_dataset):
    m = len(expert_dataset["trajectories"])
    expert_fev = torch.zeros(env.action_space.n*env.observation_space.n)
    for traj in expert_dataset["trajectories"]:
        for t,pair in enumerate(traj):
            state, action, next_state,_,_ = pair
            state = torch.where(state)
            action = torch.where(action)
            expert_fev += gamma**t*phi(torch.cat([state[0],action[0]], axis=0))
    expert_fev /= m
    expert_fev *= (1 - gamma)
    return expert_fev

expert_fev = compute_expert_fev(expert_dataset)

class Policy:
    def __init__(self, theta, phi, alpha):
        self.phi = phi
        self.alpha = alpha
        self.theta_sum = theta
        self.old_increment = 0
        
    def get_probabilities(self, s):
        deltas = torch.tensor([self.alpha*(self.phi(torch.tensor([s,a])).dot(self.theta_sum)) for a in range(env.action_space.n)])
        exp_deltas = - deltas + np.log(env.action_space.n)
        return torch.softmax(exp_deltas, axis=0)
    
    def update(self, increment):
        self.theta_sum += increment

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
            s_new,_,done,_ = env.step(a)
            counter += 1
            traj.append((s,a,s_new))
            s = s_new
        batch.append(traj)
        done=False
    return batch, counter

def sample_geometric_batch(policy, N, counter):
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
            not_stop = np.random.choice([True, False], p=[gamma, 1-gamma])
            if not not_stop:
                traj.append((s,a,s_new))
                break
            s = s_new
        if done and not_stop:
            traj.append((s,a,s_new))
        batch.append(traj)
        done=False
    return batch, counter
            
def compute_fev(batch):
    m = len(batch)
    fev = torch.zeros(1,env.action_space.n*env.observation_space.n)
    for traj in batch:
        for t,pair in enumerate(traj):
            state, action, next_state = pair
            fev += gamma**t*phi(torch.tensor([state, action]))
    fev /= m
    fev *= (1 - gamma)
    return fev

def compute_z( w, batch, theta, policy, eta = 10, alpha=0.0001):
    deltas = []
    n_trajs = 0
    for traj in batch:
        n_trajs += 1
        for _,pair in enumerate(traj):
            state, action, next_state = pair
            state = torch.tensor(state).unsqueeze(0)
            next_state = torch.tensor(next_state).unsqueeze(0)
            r = policy.phi(torch.tensor([state,action])).squeeze(0).dot(w)
            Q = policy.phi(torch.tensor([state,action])).squeeze(0).dot(theta)
            exp_V = torch.tensor([policy.phi(torch.tensor([next_state,a])).squeeze(0).dot(policy.theta_sum + theta) for a in range(env.action_space.n)])
            V = -1/alpha*torch.logsumexp(- alpha*exp_V + np.log(env.action_space.n), axis=0)
            deltas.append(torch.tensor([-r - gamma*V + Q]))
    return torch.softmax(eta*torch.cat(deltas), axis=0) #exp_deltas/torch.sum(exp_deltas)

def compute_lambda( w, batch, theta, policy, eta = 10, alpha=0.0001):
    deltas = []
    feats = []
    n_trajs = 0
    for traj in batch:
        n_trajs += 1
        for _,pair in enumerate(traj):
            state, action, next_state = pair
            state = torch.tensor(state).unsqueeze(0)
            next_state = torch.tensor(next_state).unsqueeze(0)
            feature = policy.phi(torch.tensor([state,action]))
            r = feature.squeeze(0).dot(w)
            Q = feature.squeeze(0).dot(theta)
            exp_V = torch.tensor([policy.phi(torch.tensor([next_state,a])).squeeze(0).dot(policy.theta_sum + theta) for a in range(env.action_space.n)])
            V = -1/alpha*torch.logsumexp(- alpha*exp_V + np.log(env.action_space.n), axis=0)
            deltas.append(torch.tensor([-r - gamma*V + Q]))
            feats.append(feature)
    return torch.softmax(eta*torch.cat(deltas), axis=0).matmul(torch.stack(feats)) #exp_deltas/torch.sum(exp_deltas)
            
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

def run(expert_fev, optimizer, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    w = torch.zeros(env.action_space.n*env.observation_space.n)
    theta = torch.zeros(env.action_space.n*env.observation_space.n)
    alpha=1
    policy = Policy(theta, phi, alpha)
    
    returns = []
    steps = []
    
    counter = 0
    G = estimate_return(policy, 10)
    returns.append(G[0])
    steps.append(counter)
    for _ in range(args.K):
        n_trajs=args.n_trajs
        batch, counter = sample_geometric_batch(policy, n_trajs, counter)
        batch_flatten = []
        for traj in batch:
            for _,pair in enumerate(traj):
                batch_flatten.append(pair)
        if optimizer=="forb":
            old_gradient_theta = 0
            old_gradient_w = 0
        if optimizer=="adam":
            opt_w = AdamOptimizer(env.action_space.n*env.observation_space.n, lr=args.learning_rate_w)
            opt_theta = AdamOptimizer(env.action_space.n*env.observation_space.n, lr=args.learning_rate_theta)
    
        
        for _ in range(args.T): #20
            z = compute_z(w,batch, theta, policy, alpha=alpha)
            
            # Sampling S, A, S^prime, A^prime
            distr = torch.distributions.Categorical(z)
            index = distr.sample()
            state_t, action_t, next_state_t = batch_flatten[index]
            
            probs = policy.get_probabilities(torch.tensor(next_state_t))
            distr = torch.distributions.Categorical(probs)
            next_action_t = distr.sample()
            
            #Sampling tilde S,A
            state_0 = reset_env.reset()
            probs = policy.get_probabilities(torch.tensor(state_0))
            distr = torch.distributions.Categorical(probs)
            action_0 = distr.sample()
            
            #Compute theta gradient
            gradient_theta_1 = gamma*policy.phi(torch.tensor([next_state_t, next_action_t]))
            gradient_theta_2 = - policy.phi(torch.tensor([state_t, action_t]))
            gradient_theta_3 = (1-gamma)*policy.phi(torch.tensor([state_0, action_0]))
            gradient_theta = gradient_theta_1 + gradient_theta_2 + gradient_theta_3
            if optimizer == "adam":
                theta += opt_theta.update(gradient_theta.squeeze(0))
            elif optimizer == "forb":
                theta = theta + args.learning_rate_theta*(2*gradient_theta.squeeze(0) - old_gradient_theta)
            elif optimizer == "sgd":
                theta += args.learning_rate_theta*gradient_theta.squeeze(0)

            #Compute w gradient
            distr = torch.distributions.Categorical(z)
            index = distr.sample()
            state_t, action_t, _ = batch_flatten[index]
            gradient_w = policy.phi(torch.tensor([state_t, action_t])) - expert_fev
            if optimizer == "adam":
                w += opt_w.update(gradient_w.squeeze(0)) #
            elif optimizer == "forb":
                w += args.learning_rate_w*(2*gradient_w.squeeze(0) - old_gradient_w)
            elif optimizer == "sgd":
                w +=args.learning_rate_w*gradient_w.squeeze(0)
            
            if torch.sqrt(torch.sum(w**2)) > 10:
                w /= (torch.sqrt(torch.sum(w**2))/10)
            if torch.sqrt(torch.sum(theta**2)) > 10:
                theta/= (torch.sqrt(torch.sum(theta**2))/10)
            if optimizer=="forb":
                old_gradient_theta = gradient_theta.squeeze(0)
                old_gradient_w = gradient_w.squeeze(0)

        policy.update(theta)

        G = estimate_return(policy, 10)
        returns.append(G[0])
        steps.append(counter)
        
        print(counter, G)
    with open("results/"+args.env_name+str(seed)
                +args.optimizer+"n_trajs"+str(args.n_trajs)
                +"_lr_theta"+str(args.learning_rate_theta)
                +"_lr_w"+str(args.learning_rate_w)+".pt","wb") as f:
        pickle.dump((returns, steps), f)
    return policy, w

def run_primal_dual(expert_fev, optimizer, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    w = torch.zeros(env.action_space.n*env.observation_space.n)
    theta = torch.zeros(env.action_space.n*env.observation_space.n)
    alpha=1
    policy = Policy(theta, phi, alpha)
    
    returns = []
    steps = []
    
    counter = 0
    G = estimate_return(policy, 10)
    returns.append(G[0])
    steps.append(counter)
    for _ in range(args.K):
        n_trajs=args.n_trajs
        batch, counter = sample_geometric_batch(policy, n_trajs, counter)
        batch_flatten = []
        for traj in batch:
            for _,pair in enumerate(traj):
                batch_flatten.append(pair)
        
        if optimizer=="forb":
            old_gradient_theta = 0
            old_gradient_w = 0
        if optimizer=="adam":
            opt_w = AdamOptimizer(env.action_space.n*env.observation_space.n, lr=args.learning_rate_w)
            opt_theta = AdamOptimizer(env.action_space.n*env.observation_space.n, lr=args.learning_rate_theta)

        for _ in range(args.T):
            z = compute_z(w,batch, theta, policy,alpha=alpha)
            
            # Sampling S, A, S^prime, A^prime
            distr = torch.distributions.Categorical(z)
            index = distr.sample() #np.random.choice(len(z), p=z)
            state_t, action_t, next_state_t = batch_flatten[index]
            
            probs = policy.get_probabilities(torch.tensor(next_state_t))
            distr = torch.distributions.Categorical(probs)
            next_action_t = distr.sample()
            
            #Sampling tilde S,A
            state_0 = reset_env.reset()
            probs = policy.get_probabilities(torch.tensor(state_0))
            distr = torch.distributions.Categorical(probs)
            action_0 = distr.sample()
            
            #Compute theta gradient
            gradient_theta_1 = gamma*policy.phi(torch.tensor([next_state_t, next_action_t]))
            gradient_theta_2 = - policy.phi(torch.tensor([state_t, action_t]))
            gradient_theta_3 = (1-gamma)*policy.phi(torch.tensor([state_0, action_0]))
            gradient_theta = gradient_theta_1 + gradient_theta_2 + gradient_theta_3
            if optimizer == "adam":
                theta += opt_theta.update(gradient_theta.squeeze(0))
            elif optimizer == "forb":
                theta = theta + args.learning_rate_theta*(2*gradient_theta.squeeze(0) - old_gradient_theta)
            elif optimizer == "sgd":
                theta += args.learning_rate_theta*gradient_theta.squeeze(0)
            if optimizer == "forb":
                old_gradient_theta = gradient_theta.squeeze(0)
            
            if torch.sqrt(torch.sum(theta**2)) > 10:
                #print("Projecting theta")
                theta/= (torch.sqrt(torch.sum(theta**2))/10)
        
        policy.update(theta)
        n_dual_updates=args.n_dual_updates
        for _ in range(n_dual_updates):
            #Compute w gradient
            batch, counter = sample_geometric_batch(policy, 1, counter)
            batch_flatten = []
            for traj in batch:
                for _,pair in enumerate(traj):
                    batch_flatten.append(pair)
            state_t, action_t, _ = batch_flatten[0]
            gradient_w = policy.phi(torch.tensor([state_t, action_t])) - expert_fev
            if optimizer == "adam":
                w += opt_w.update(gradient_w.squeeze(0)) #
            elif optimizer == "forb":
                w += args.learning_rate_w*(2*gradient_w.squeeze(0) - old_gradient_w)
            elif optimizer == "sgd":
                w +=args.learning_rate_w*gradient_w.squeeze(0)
            if torch.sqrt(torch.sum(w**2)) > 10:
                #print("Projecting w")
                w /= (torch.sqrt(torch.sum(w**2))/10)
            old_gradient_w = gradient_w.squeeze(0)

        G = estimate_return(policy, 10)
        returns.append(G[0])
        steps.append(counter)
        
        print(counter, G)
    with open("results/primal_dual"+args.env_name+str(seed)
                +args.optimizer+"n_trajs"+str(args.n_trajs)
                +"_lr_theta"+str(args.learning_rate_theta)
                +"_lr_w"+str(args.learning_rate_w)+".pt","wb") as f:
        import pickle
        pickle.dump((returns, steps), f)
    return policy, w

"""print("seed", args.seed)
print("_______________________________________________")
if not args.primal_dual:
    policy = run(expert_fev, optimizer=args.optimizer, seed = args.seed)
else:
    policy = run_primal_dual(expert_fev, optimizer=args.optimizer, seed = args.seed)"""


def run_incremental(expert_fev, optimizer, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    w = torch.zeros(env.action_space.n*env.observation_space.n)
    theta = torch.zeros(env.action_space.n*env.observation_space.n)
    alpha=1
    policy = Policy(theta, phi, alpha)
    
    returns = []
    steps = []
    
    counter = 0
    G = estimate_return(policy, 10)
    returns.append(G[0])
    steps.append(counter)

    if optimizer=="forb":
        old_gradient_theta = 0
        old_gradient_w = 0
    if optimizer=="adam":
        opt_w = AdamOptimizer(env.action_space.n*env.observation_space.n, lr=args.learning_rate_w)
        opt_theta = AdamOptimizer(env.action_space.n*env.observation_space.n, lr=args.learning_rate_theta)
    for _ in range(args.K):
        n_trajs=args.n_trajs
        batch, counter = sample_geometric_batch(policy, n_trajs, counter)
        batch_flatten = []
        for traj in batch:
            for _,pair in enumerate(traj):
                batch_flatten.append(pair)

        total_sample= len(batch_flatten)
        gradient_theta = 0
        
        for (state_t, action_t, _) in batch_flatten:
            gradient_theta += policy.phi(torch.tensor([state_t, action_t]))/total_sample
        optimal_lambda = compute_lambda(w,batch, theta, policy,alpha=alpha)
        
        gradient_theta -= optimal_lambda 

        #Sampling tilde S,A
        state_0 = reset_env.reset()
        probs = policy.get_probabilities(torch.tensor(state_0))
        distr = torch.distributions.Categorical(probs)
        action_0 = distr.sample()
            
        gradient_theta += (1-gamma)*policy.phi(torch.tensor([state_0, action_0]))

        if optimizer == "eg":
            old_theta = theta
        if optimizer == "adam":
            theta += opt_theta.update(gradient_theta.squeeze(0))
        elif optimizer == "forb":
            theta = theta + args.learning_rate_theta*(2*gradient_theta.squeeze(0) - old_gradient_theta)
        elif optimizer == "sgd" or optimizer == "eg":
            theta += args.learning_rate_theta*gradient_theta.squeeze(0)
        if optimizer == "forb":
            old_gradient_theta = gradient_theta.squeeze(0)
        
        n_dual_updates=args.n_dual_updates
        for _ in range(n_dual_updates):
            #Compute w gradient
            batch, counter = sample_geometric_batch(policy, n_trajs, counter)
            batch_flatten = []
            for traj in batch:
                for _,pair in enumerate(traj):
                    batch_flatten.append(pair)
            #state_t, action_t, _ = batch_flatten[0]
            #gradient_w = policy.phi(torch.tensor([state_t, action_t])) - expert_fev
            #total_sample= len(batch_flatten)
            #gradient_w = 0
            #for (state_t, action_t, _) in batch_flatten:
            #    gradient_w += policy.phi(torch.tensor([state_t, action_t]))/total_sample
            #gradient_w =  - expert_fev
            optimal_lambda = compute_lambda(w,batch, theta, policy,alpha=alpha)
            gradient_w = optimal_lambda - expert_fev
            if optimizer == "eg":
                old_w = w
            if optimizer == "adam":
                w += opt_w.update(gradient_w.squeeze(0)) #
            elif optimizer == "forb":
                w += args.learning_rate_w*(2*gradient_w.squeeze(0) - old_gradient_w)
            elif optimizer == "sgd" or optimizer == "eg":
                w +=args.learning_rate_w*gradient_w.squeeze(0)
            if torch.sqrt(torch.sum(w**2)) > 10:
                print("Projecting w")
                w /= (torch.sqrt(torch.sum(w**2))/10)
            old_gradient_w = gradient_w.squeeze(0)
        policy.update(theta)
        G = estimate_return(policy, 10)
        returns.append(G[0])
        steps.append(counter)
        if optimizer=="eg":
            batch, counter = sample_geometric_batch(policy, n_trajs, counter)
            batch_flatten = []
            for traj in batch:
                for _,pair in enumerate(traj):
                    batch_flatten.append(pair)

            total_sample= len(batch_flatten)
            gradient_theta = 0
            
            for (state_t, action_t, _) in batch_flatten:
                gradient_theta += policy.phi(torch.tensor([state_t, action_t]))/total_sample
            optimal_lambda = compute_lambda(w,batch, theta, policy,alpha=alpha)
            
            gradient_theta -= optimal_lambda 
            #Sampling tilde S,A
            state_0 = reset_env.reset()
            probs = policy.get_probabilities(torch.tensor(state_0))
            distr = torch.distributions.Categorical(probs)
            action_0 = distr.sample()
                
            gradient_theta += (1-gamma)*policy.phi(torch.tensor([state_0, action_0]))
            new_theta = old_theta + args.learning_rate_theta*gradient_theta.squeeze(0)
            
            for _ in range(n_dual_updates):
                #Compute w gradient
                batch, counter = sample_geometric_batch(policy, n_trajs, counter)
                batch_flatten = []
                for traj in batch:
                    for _,pair in enumerate(traj):
                        batch_flatten.append(pair)
                #state_t, action_t, _ = batch_flatten[0]
                #gradient_w = policy.phi(torch.tensor([state_t, action_t])) - expert_fev
                #total_sample= len(batch_flatten)
                #gradient_w = 0
                #for (state_t, action_t, _) in batch_flatten:
                #    gradient_w += policy.phi(torch.tensor([state_t, action_t]))/total_sample
                #gradient_w =  - expert_fev
                optimal_lambda = compute_lambda(w,batch, theta, policy,alpha=alpha)
                gradient_w = optimal_lambda - expert_fev
                w = old_w + args.learning_rate_w*gradient_w.squeeze(0)
                if torch.sqrt(torch.sum(w**2)) > 10:
                    print("Projecting w")
                    w /= (torch.sqrt(torch.sum(w**2))/10)
        policy.update(new_theta-theta)
        theta = new_theta
        print(counter, G)
    with open("results/incremental"+args.env_name+str(seed)
                +args.optimizer+"n_trajs"+str(args.n_trajs)
                +"_lr_theta"+str(args.learning_rate_theta)
                +"_lr_w"+str(args.learning_rate_w)+".pt","wb") as f:
        import pickle
        pickle.dump((returns, steps), f)
    return policy, w

def run_incremental_biased_sgd(expert_fev, optimizer, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    w = torch.zeros(env.action_space.n*env.observation_space.n)
    theta = torch.zeros(env.action_space.n*env.observation_space.n)
    lambda_var = torch.ones(env.action_space.n*env.observation_space.n)/(env.action_space.n*env.observation_space.n) #TODO change size for other features
    alpha=1
    ridge_par=1e-3
    policy = Policy(theta, phi, alpha)
    
    returns = []
    steps = []
    
    counter = 0
    G = estimate_return(policy, 10)
    returns.append(G[0])
    steps.append(counter)

    if optimizer=="forb":
        old_gradient_theta = 0
        old_gradient_w = 0
    if optimizer=="adam":
        opt_w = AdamOptimizer(env.action_space.n*env.observation_space.n, lr=args.learning_rate_w)
        opt_theta = AdamOptimizer(env.action_space.n*env.observation_space.n, lr=args.learning_rate_theta)
    for k in range(args.K):
        learning_rate_theta = args.learning_rate_theta/((k+1)**(1/3))
        learning_rate_w = args.learning_rate_w/((k+1)**(1/3))
        n_trajs=np.int(args.n_trajs*(k+1)**(2/3))
        print(n_trajs)
        batch, counter = sample_geometric_batch(policy, n_trajs, counter)
        batch_flatten = []
        for traj in batch:
            for _,pair in enumerate(traj):
                batch_flatten.append(pair)

        total_sample= len(batch_flatten)
        gradient_theta = 0
        
        for (state_t, action_t, _) in batch_flatten:
            gradient_theta += policy.phi(torch.tensor([state_t, action_t]))/total_sample
        #optimal_lambda = compute_lambda(w,batch, theta, policy,alpha=alpha)
        sum_phi=0
        weighted_sum=0
        for state_t, action_t, next_state_t in batch_flatten:
            sum_phi += policy.phi(torch.tensor([state_t, action_t])).reshape(-1,1).matmul(policy.phi(torch.tensor([state_t, action_t])).reshape(1,-1))

            exp_V = torch.tensor([policy.phi(torch.tensor([next_state_t,a])).squeeze(0).dot(policy.theta_sum + theta) for a in range(env.action_space.n)])
            V = -1/alpha*torch.logsumexp(- alpha*exp_V + np.log(env.action_space.n), axis=0)
            weighted_sum += policy.phi(torch.tensor([state_t, action_t]))*V
        
        MV_est = np.linalg.solve(sum_phi + ridge_par*np.eye(env.observation_space.n*env.action_space.n), weighted_sum)
        delta = w + gamma*MV_est - theta
        

        
        gradient_theta -= lambda_var

        #Sampling tilde S,A
        state_0 = reset_env.reset()
        probs = policy.get_probabilities(torch.tensor(state_0))
        distr = torch.distributions.Categorical(probs)
        action_0 = distr.sample()
            
        gradient_theta += (1-gamma)*policy.phi(torch.tensor([state_0, action_0]))

        if optimizer == "eg":
            old_theta = theta
            old_lambda_var = lambda_var
        if optimizer == "adam":
            theta += opt_theta.update(gradient_theta.squeeze(0))
        elif optimizer == "forb":
            theta = theta + args.learning_rate_theta*(2*gradient_theta.squeeze(0) - old_gradient_theta)
        elif optimizer == "sgd" or optimizer == "eg":
            theta += learning_rate_theta*gradient_theta.squeeze(0)
            lambda_var = torch.softmax(-20/((k+1)**(1/3))*delta + torch.log(lambda_var),0)
        if optimizer == "forb":
            old_gradient_theta = gradient_theta.squeeze(0)
        
        #n_dual_updates=args.n_dual_updates
        #for _ in range(n_dual_updates):
        #Compute w gradient
        #batch, counter = sample_geometric_batch(policy, n_trajs, counter)
        #batch_flatten = []
        #for traj in batch:
        #    for _,pair in enumerate(traj):
        #        batch_flatten.append(pair)
        #state_t, action_t, _ = batch_flatten[0]
        #gradient_w = policy.phi(torch.tensor([state_t, action_t])) - expert_fev
        #total_sample= len(batch_flatten)
        #gradient_w = 0
        #for (state_t, action_t, _) in batch_flatten:
        #    gradient_w += policy.phi(torch.tensor([state_t, action_t]))/total_sample
        #gradient_w =  - expert_fev
        #optimal_lambda = compute_lambda(w,batch, theta, policy,alpha=alpha)
        gradient_w = lambda_var - expert_fev
        if optimizer == "eg":
            old_w = w
        if optimizer == "adam":
            w += opt_w.update(gradient_w.squeeze(0)) #
        elif optimizer == "forb":
            w += args.learning_rate_w*(2*gradient_w.squeeze(0) - old_gradient_w)
        elif optimizer == "sgd" or optimizer == "eg":
            w +=learning_rate_w*gradient_w.squeeze(0)
        if torch.sqrt(torch.sum(w**2)) > 10:
            print("Projecting w")
            w /= (torch.sqrt(torch.sum(w**2))/10)
        old_gradient_w = gradient_w.squeeze(0)
        policy.update(theta)
        G = estimate_return(policy, 10)
        returns.append(G[0])
        steps.append(counter)
        if optimizer=="eg":
            batch, counter = sample_geometric_batch(policy, n_trajs, counter)
            batch_flatten = []
            for traj in batch:
                for _,pair in enumerate(traj):
                    batch_flatten.append(pair)

            total_sample= len(batch_flatten)
            gradient_theta = 0
            
            for (state_t, action_t, _) in batch_flatten:
                gradient_theta += policy.phi(torch.tensor([state_t, action_t]))/total_sample
            #optimal_lambda = compute_lambda(w,batch, theta, policy,alpha=alpha)
            sum_phi=0
            weighted_sum=0
            for state_t, action_t, next_state_t in batch_flatten:
                sum_phi += policy.phi(torch.tensor([state_t, action_t])).reshape(-1,1).matmul(policy.phi(torch.tensor([state_t, action_t])).reshape(1,-1))

                exp_V = torch.tensor([policy.phi(torch.tensor([next_state_t,a])).squeeze(0).dot(policy.theta_sum + theta) for a in range(env.action_space.n)])
                V = -1/alpha*torch.logsumexp(- alpha*exp_V + np.log(env.action_space.n), axis=0)
                weighted_sum += policy.phi(torch.tensor([state_t, action_t]))*V
            
            MV_est = np.linalg.solve(sum_phi + ridge_par*np.eye(env.observation_space.n*env.action_space.n), weighted_sum)
            delta = w + gamma*MV_est - theta
            
            gradient_theta -= lambda_var
            #Sampling tilde S,A
            state_0 = reset_env.reset()
            probs = policy.get_probabilities(torch.tensor(state_0))
            distr = torch.distributions.Categorical(probs)
            action_0 = distr.sample()
                
            gradient_theta += (1-gamma)*policy.phi(torch.tensor([state_0, action_0]))
            new_theta = old_theta + args.learning_rate_theta*gradient_theta.squeeze(0)
            
            #for _ in range(n_dual_updates):
            #Compute w gradient
            #    batch, counter = sample_geometric_batch(policy, n_trajs, counter)
            #    batch_flatten = []
            #    for traj in batch:
            #        for _,pair in enumerate(traj):
            #            batch_flatten.append(pair)
            #state_t, action_t, _ = batch_flatten[0]
            #gradient_w = policy.phi(torch.tensor([state_t, action_t])) - expert_fev
            #total_sample= len(batch_flatten)
            #gradient_w = 0
            #for (state_t, action_t, _) in batch_flatten:
            #    gradient_w += policy.phi(torch.tensor([state_t, action_t]))/total_sample
            #gradient_w =  - expert_fev
            #optimal_lambda = compute_lambda(w,batch, theta, policy,alpha=alpha)

            gradient_w = lambda_var - expert_fev
            w = old_w + args.learning_rate_w*gradient_w.squeeze(0)
            if torch.sqrt(torch.sum(w**2)) > 10:
                print("Projecting w")
                w /= (torch.sqrt(torch.sum(w**2))/10)
            lambda_var = torch.softmax(-20/((k+1)**(1/3))*delta + torch.log(old_lambda_var),0)
        policy.update(new_theta-theta)
        theta = new_theta
        print(counter, G)
    with open("results/incremental_biased_sgd"+args.env_name+str(seed)
                +args.optimizer+"n_trajs"+str(args.n_trajs)
                +"_lr_theta"+str(args.learning_rate_theta)
                +"_lr_w"+str(args.learning_rate_w)+".pt","wb") as f:
        import pickle
        pickle.dump((returns, steps), f)
    return policy, w

def run_biased_sgd(expert_fev, optimizer, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    w = torch.zeros(env.action_space.n*env.observation_space.n)
    theta = torch.zeros(env.action_space.n*env.observation_space.n)
    alpha=1
    eta=10
    policy = Policy(theta, phi, alpha)
    
    returns = []
    steps = []
    ridge_par=1e-3
    counter = 0
    G = estimate_return(policy, 10)
    returns.append(G[0])
    steps.append(counter)
    for k in range(args.K):
        n_trajs=np.int(args.n_trajs*np.sqrt(k+1))
        
        if optimizer=="forb":
            old_gradient_theta = 0
            old_gradient_w = 0
        if optimizer=="adam":
            opt_w = AdamOptimizer(env.action_space.n*env.observation_space.n, lr=args.learning_rate_w)
            opt_theta = AdamOptimizer(env.action_space.n*env.observation_space.n, lr=args.learning_rate_theta)

        
        batch, counter = sample_geometric_batch(policy, n_trajs, counter)
        batch_flatten = []
        for traj in batch:
            for _,pair in enumerate(traj):
                batch_flatten.append(pair)
        sum_phi=0
        weighted_sum=0
        #Biased Gradient estimation
        for t in range(args.T): #20
            learning_rate_theta = args.learning_rate_theta / np.sqrt(t+1)
            learning_rate_w = args.learning_rate_w / np.sqrt(t+1)
            for state_t, action_t, next_state_t in batch_flatten:
                sum_phi += policy.phi(torch.tensor([state_t, action_t])).reshape(-1,1).matmul(policy.phi(torch.tensor([state_t, action_t])).reshape(1,-1))

                exp_V = torch.tensor([policy.phi(torch.tensor([next_state_t,a])).squeeze(0).dot(policy.theta_sum + theta) for a in range(env.action_space.n)])
                V = -1/alpha*torch.logsumexp(- alpha*exp_V + np.log(env.action_space.n), axis=0)
                weighted_sum += policy.phi(torch.tensor([state_t, action_t]))*V
            
            MV_est = np.linalg.solve(sum_phi + ridge_par*np.eye(env.observation_space.n*env.action_space.n), weighted_sum)

            delta = w + gamma*MV_est - theta
            #phi_hat = torch.diag(sum_phi)/len(batch_flatten) + ridge_par
            #B_est = torch.softmax(-eta/np.sqrt(k+1)*delta + torch.log(phi_hat),0) / phi_hat
            B_est = torch.softmax(-eta*delta,0)
            #Sampling tilde S,A
            state_0 = reset_env.reset()
            probs = policy.get_probabilities(torch.tensor(state_0))
            distr = torch.distributions.Categorical(probs)
            action_0 = distr.sample()

            batch, counter = sample_geometric_batch(policy, 1, counter)
            batch_flatten = []
            for traj in batch:
                for _,pair in enumerate(traj):
                    batch_flatten.append(pair)
            state_t, action_t, next_state_t = batch_flatten[0]
            probs = policy.get_probabilities(torch.tensor(next_state_t))
            distr = torch.distributions.Categorical(probs)
            next_action_t = distr.sample()

            gradient_theta = (1-gamma)*policy.phi(torch.tensor([state_0, action_0])) \
                + B_est[state_t*env.action_space.n + action_t]\
                    *(gamma*policy.phi(torch.tensor([next_state_t, next_action_t])) - policy.phi(torch.tensor([state_t, action_t])))
            
            gradient_w = B_est[state_t*env.action_space.n + action_t]*policy.phi(torch.tensor([state_t, action_t])) - expert_fev
            
            if optimizer == "adam":
                theta += opt_theta.update(gradient_theta.squeeze(0))
            elif optimizer == "forb":
                theta = theta + learning_rate_theta*(2*gradient_theta.squeeze(0) - old_gradient_theta)
            elif optimizer == "sgd":
                theta += args.learning_rate_theta*gradient_theta.squeeze(0)

            if optimizer == "adam":
                w += opt_w.update(gradient_w.squeeze(0)) #
            elif optimizer == "forb":
                w += learning_rate_w*(2*gradient_w.squeeze(0) - old_gradient_w)
            elif optimizer == "sgd":
                w +=args.learning_rate_w*gradient_w.squeeze(0)
            
            if torch.sqrt(torch.sum(w**2)) > 10:
                w /= (torch.sqrt(torch.sum(w**2))/10)
            if torch.sqrt(torch.sum(theta**2)) > 10:
                theta/= (torch.sqrt(torch.sum(theta**2))/10)
            if optimizer=="forb":
                old_gradient_theta = gradient_theta.squeeze(0)
                old_gradient_w = gradient_w.squeeze(0)

        policy.update(theta)

        G = estimate_return(policy, 10)
        returns.append(G[0])
        steps.append(counter)
        
        print(counter, G)
    with open("results/reuse_biased_sgd"+args.env_name+str(seed)
                +args.optimizer+"n_trajs"+str(args.n_trajs)
                +"_lr_theta"+str(args.learning_rate_theta)
                +"_lr_w"+str(args.learning_rate_w)+".pt","wb") as f:
        pickle.dump((returns, steps), f)
    return policy, w

print("seed", args.seed)
print("_______________________________________________")

if (not args.primal_dual and not args.incremental and not args.biased_sgd):
    policy = run(expert_fev, optimizer=args.optimizer, seed = args.seed)
elif args.incremental:
    policy = run_incremental_biased_sgd(expert_fev, optimizer=args.optimizer, seed = args.seed)
elif args.biased_sgd:
    policy = run_biased_sgd(expert_fev, optimizer=args.optimizer, seed = args.seed)
else:
    policy = run_primal_dual(expert_fev, optimizer=args.optimizer, seed = args.seed)


