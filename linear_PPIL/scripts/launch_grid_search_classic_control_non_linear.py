import os
import argparse

parser = argparse.ArgumentParser(description='Grid search hyperparameters')

parser.add_argument('--env-name', metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-traj-path', metavar='G',
                    help='path of the expert trajectories')
parser.add_argument('--primal-dual', action="store_true", default=False)
args = parser.parse_args()
index = 0
for lr_w in [5e-1, 1e-1, 5e-2, 1e-2]:
    for lr_theta in [5e-3, 3e-3, 1e-3, 7e-4, 6e-4, 5e-1, 1e-1, 5e-2, 1e-2]:
        for seed in range(10):
            for n_trajs in [50, 100, 200]:
                for optimizer in ["sgd", "forb"]:
                    string = f"python run_classic_control_non_linear.py --env-name {args.env_name} \
                        --expert-traj-path {args.expert_traj_path} \
                            --learning-rate-w {lr_w} --learning-rate-theta {lr_theta} \
                                --seed {seed} --n-trajs {n_trajs} \
                                    --K 50 --T 20 --optimizer {optimizer}"
                    command = f"command{args.env_name}_lr_w{lr_w}_lr_theta{lr_theta}seed_{seed}n_trajs{n_trajs}optimizer{optimizer}"
                
                    if args.primal_dual:
                        string = f"{string} --primal-dual"
                        command = f"{command}primal_dual"
                    command=f"{command}.txt"
                    with open(command, 'w') as file:
                        file.write(f'{string}\n')
                    
                    os.system(f'sbatch --job-name={index} submit.sh {command}')

                    index +=1
