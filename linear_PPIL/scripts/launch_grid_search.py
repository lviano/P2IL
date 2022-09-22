import os
import argparse

parser = argparse.ArgumentParser(description='Grid search hyperparameters')

parser.add_argument('--env-name', metavar='G',
                    help='name of the environment to run')
parser.add_argument('--expert-traj-path', metavar='G',
                    help='path of the expert trajectories')
parser.add_argument('--primal-dual', action="store_true", default=False)
parser.add_argument('--incremental', action="store_true", default=False)
parser.add_argument('--biased-sgd', action="store_true", default=False)
args = parser.parse_args()
index = 0
for lr_w in [5e-1, 3e-1, 1e-1]:
    for lr_theta in [5e-3, 3e-3, 1e-3, 7e-4, 6e-4, 5e-1, 1e-1, 5e-2, 1e-2]:
        for seed in range(50):
            for n_trajs in [25,50]:
                for optimizer in ["forb"]: #"adam", "eg"]:
                    string = f"python run.py --env-name {args.env_name} \
                        --expert-traj-path {args.expert_traj_path} \
                            --learning-rate-w {lr_w} --learning-rate-theta {lr_theta} \
                                --seed {seed} --n-trajs {n_trajs} \
                                    --K 25 --T 20 --optimizer {optimizer} --n-dual-updates 1"
                    command = f"command{args.env_name}_lr_w{lr_w}_lr_theta{lr_theta}seed_{seed}n_trajs{n_trajs}optimizer{optimizer}"
                
                    if args.primal_dual:
                        string = f"{string} --primal-dual"
                        command = f"{command}primal_dual"
                    if args.incremental:
                        string = f"{string} --incremental"
                        command = f"{command}incremental"
                    if args.biased_sgd:
                        string = f"{string} --biased-sgd"
                        command = f"{command}biasedsgd"
                    command=f"{command}.txt"
                    with open(command, 'w') as file:
                        file.write(f'{string}\n')
                    
                    os.system(f'sbatch --job-name={index} submit.sh {command}')

                    index +=1
