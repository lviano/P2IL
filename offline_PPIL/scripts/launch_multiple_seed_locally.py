import os
import argparse

parser = argparse.ArgumentParser(description='Grid search hyperparameters')

parser.add_argument('--env-name', metavar='G',
                    help='name of the environment to run')
args = parser.parse_args()
for env in ["cartpole","acrobot","lunarlander"]:
    for method in ["logistic_offline"]:
        for eval_demos in [1,3,5,7,10, 15]:
            for seed in range(5,10):
                string = f"python train_iq.py agent=softq env={env} \
                    eval.demos={eval_demos} eval.subsample_freq=5 \
                        method.loss=value seed={seed} method.type={method} agent.init_temperature=0.01" #temp 1 for acrobot and cartpole    
                #os.system(string)
                command = f"command{seed}{env}LOoffline{eval_demos}"
                command=f"{command}.txt"
                with open(command, 'w') as file:
                    file.write(f'{string}\n')
                
                os.system(f'sbatch --job-name=LO{seed} submit.sh {command}')
