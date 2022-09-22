import os
import argparse

parser = argparse.ArgumentParser(description='Grid search hyperparameters')

parser.add_argument('--env-name', metavar='G',
                    help='name of the environment to run')
args = parser.parse_args()
for env in ["cheetah","hopper"]:
    for method in ["logistic", "logistic_primal_dual"]:
        for actor_lr in [3e-3, 3e-2, 3e-4]:
            for critic_lr in [3e-3, 3e-2, 3e-4]:
                for alpha in [10, 1, 1e-1, 1e-2]:
                    for seed in range(1):
                        string = f"python train_iq.py agent=sac env={env} \
                            eval.demos=10 \
                                method.loss=value seed={seed} \
                                    method.type={method} \
                                        method.regularize=False \
                                            agent.actor_lr={actor_lr} \
                                                agent.critic_lr={critic_lr} \
                                                    agent.init_temperature=1"
                        command = f"command{seed}{env}{method}"
                        command=f"{command}.txt"
                        with open(command, 'w') as file:
                            file.write(f'{string}\n')
                        
                        os.system(f'sbatch --job-name={env}{method}{seed} submit.sh {command}')