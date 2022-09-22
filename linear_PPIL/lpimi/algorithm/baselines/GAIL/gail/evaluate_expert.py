import argparse
import gym
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../external/scuri-rllib/')))
from itertools import count
from utils import *
from rllib.environment import GymEnvironment
from lpimi.envs.wrappers import TimeLimit

parser = argparse.ArgumentParser(description='Save expert trajectory')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--model-path', metavar='G',
                    help='name of the expert model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--max-expert-state-num', type=int, default=50000, metavar='N',
                    help='maximal number of main iterations (default: 50000)')
args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)

if args.env_name in ["DeepSea-v0", "WideTree-v0", "DoubleChainProblem-v0", "RiverSwim-v0", "SingleChainProblem-v0", "TwoStateStochastic-v0", "TwoStateProblem-v0", "EasyGridWorld-v0", "WindyGrid-v0"]:
    cfg={} # TODO: set configs according to name
    env = TimeLimit(GymEnvironment(args.env_name, seed=args.seed, **cfg), 200)
    state_dim = env.observation_space.n
    is_disc_state = len(env.observation_space.shape) == 0
else:
    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    is_disc_state = False
env.seed(args.seed)
torch.manual_seed(args.seed)
is_disc_action = len(env.action_space.shape) == 0

if is_disc_state:
    policy_net, _ = pickle.load(open(args.model_path, "rb"))
    running_state = lambda x: torch.eye((state_dim))[x]
    if is_disc_action:
        action_to_torch = lambda x: torch.eye((env.action_space.n))[x]
else:
    policy_net, _, running_state = pickle.load(open(args.model_path, "rb"))
    if is_disc_action:
        action_to_torch = lambda x: torch.eye((env.action_space.n))[x]
running_state.fix = True
expert_traj = []
expert_traj_lpimi = []
def main_loop():

    num_steps = 0
    episode_rewards = []
    for i_episode in count():

        state = env.reset()
        state = running_state(state)
        reward_episode = 0
        expert_traj_lpimi.append([])
        for t in range(10000):
            
            state_var = tensor(state).unsqueeze(0).to(dtype)
            # choose mean action
            #action = policy_net(state_var)[0][0].detach().numpy()
            # choose stochastic action
            action = policy_net.select_action(state_var)[0].cpu().numpy()
            action = int(action) if is_disc_action else action.astype(np.float64)
            if args.env_name == "DeepSea-v0":
                action = 1
            next_state, reward, done, _ = env.step(action)
            next_state = running_state(next_state)
            reward_episode += reward
            
            num_steps += 1

            expert_traj.append(np.hstack([state, action]))
            if is_disc_action and is_disc_state:
                expert_traj_lpimi[-1].append((state, action_to_torch(action), next_state, reward, None))
            elif is_disc_action and not is_disc_state:
                expert_traj_lpimi[-1].append((torch.tensor(state, dtype=torch.float32), torch.tensor(action_to_torch(action),dtype=torch.float32), torch.tensor(next_state, dtype=torch.float32), reward, None))
            else:
                expert_traj_lpimi[-1].append((state, torch.tensor(action), next_state, reward, None))
            if args.render:
                env.render()
            if done or num_steps >= args.max_expert_state_num:
                break

            state = next_state

        print('Episode {}\t reward: {:.2f}'.format(i_episode, reward_episode))
        episode_rewards.append(reward_episode)
        if num_steps >= args.max_expert_state_num:

            break
    print(np.mean(episode_rewards))


main_loop()
