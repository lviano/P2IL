import os
import argparse

parser = argparse.ArgumentParser(description='Grid search hyperparameters')

parser.add_argument('--env-name', metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-traj-path', metavar='G',
                    help='path of the expert trajectories')
parser.add_argument('--primal-dual', action="store_true", default=False)
parser.add_argument('--learning-rate-w', type=float, default=3e-4, metavar='G',
                    help='(default: 3e-4)')
parser.add_argument('--learning-rate-theta', type=float, default=3e-4, metavar='G',
                    help='(default: 3e-4)')
parser.add_argument('--n-trajs', type=int, default=50)
parser.add_argument('--optimizer', metavar='G', default="forb",
                    help='optimizer')
args = parser.parse_args()
for seed in range(10,50):
    string = f"python run.py --env-name {args.env_name} \
        --expert-traj-path {args.expert_traj_path} \
            --learning-rate-w {args.learning_rate_w} --learning-rate-theta {args.learning_rate_theta} \
                --seed {seed} --n-trajs {args.n_trajs} \
                    --K 50 --T 300 --optimizer {args.optimizer}"
    command = f"command{args.env_name}_lr_w{args.learning_rate_w}_lr_theta{args.learning_rate_theta}seed_{seed}n_trajs{args.n_trajs}optimizer{args.optimizer}"
        
    if args.primal_dual:
        string = f"{string} --primal-dual"
        command = f"{command}primal_dual"
    command=f"{command}.txt"
    with open(command, 'w') as file:
        file.write(f'{string}\n')
    
    os.system(f'sbatch --job-name={seed} submit.sh {command}')
