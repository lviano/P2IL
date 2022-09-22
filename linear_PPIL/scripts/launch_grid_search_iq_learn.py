import os
import argparse

parser = argparse.ArgumentParser(description='Grid search hyperparameters')

parser.add_argument('--env-name', metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-traj-path', metavar='G',
                    help='path of the expert trajectories')

args = parser.parse_args()
index = 0
for lr_theta in [5e-3, 1e-3, 5e-1, 1e-1, 5e-2, 1e-2, 9e-1]:
    for seed in range(10):
        for n_trajs in [1, 3, 5, 10]:
            for optimizer in ["forb"]:
                string = f"python run_iq_learn.py --env-name {args.env_name} \
                    --expert-traj-path {args.expert_traj_path} \
                        --learning-rate-theta {lr_theta} \
                            --seed {seed} --n-trajs {n_trajs} \
                                --K 50 --optimizer {optimizer}"
                command = f"command{args.env_name}_lr_theta{lr_theta}seed_{seed}n_trajs{n_trajs}optimizer{optimizer}IQLearn"

                command=f"{command}.txt"
                with open(command, 'w') as file:
                    file.write(f'{string}\n')
                
                os.system(f'sbatch --job-name={index}IQ submit.sh {command}')

                index +=1
