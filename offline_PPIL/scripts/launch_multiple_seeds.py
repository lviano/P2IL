import os
import argparse

parser = argparse.ArgumentParser(description='Grid search hyperparameters')

parser.add_argument('--env-name', metavar='G',
                    help='name of the environment to run')
args = parser.parse_args()
for env in ["cheetah","hopper","ant"]:
    for method in ["logistic", "logistic_primal_dual", "iq"]:
        for seed in range(1):
            string = f"python train_iq.py agent=sac env={env} \
                eval.demos=10 \
                    method.loss=value seed={seed} method.type={method} method.regularize=False agent.actor_lr=3e-3" #agent.init_temperature=1"
            command = f"command{seed}{env}{method}"
            command=f"{command}.txt"
            with open(command, 'w') as file:
                file.write(f'{string}\n')
            
            os.system(f'sbatch --job-name={env}{method}{seed} submit.sh {command}')
