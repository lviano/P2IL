import os
import argparse

parser = argparse.ArgumentParser(description='Grid search hyperparameters')

parser.add_argument('--env-name', metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-traj-path', metavar='G',
                    help='path of the expert trajectories')
parser.add_argument('--learning-rate-theta', type=float, default=0.9, metavar='G',
                    help='(default: 0.9)')
parser.add_argument('--n-trajs', type=int, default=1)
parser.add_argument('--optimizer', metavar='G', default="forb",
                    help='optimizer')
args = parser.parse_args()
for seed in range(50):
    string = f"python run_iq_learn.py --env-name {args.env_name} \
        --expert-traj-path {args.expert_traj_path} \
            --learning-rate-theta {args.learning_rate_theta} \
                --seed {seed} --n-trajs {args.n_trajs} \
                    --K 100 --optimizer {args.optimizer}"
    command = f"command{args.env_name}_lr_theta{args.learning_rate_theta}seed_{seed}n_trajs{args.n_trajs}optimizer{args.optimizer}IQLearn"
    command=f"{command}.txt"
    with open(command, 'w') as file:
        file.write(f'{string}\n')
    
    os.system(f'sbatch --job-name={seed} submit.sh {command}')