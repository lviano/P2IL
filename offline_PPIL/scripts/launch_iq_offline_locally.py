import os
import argparse

for eval_demos in [1,3,5,7,10,15]:
    for seed in range(10):
        # Acrobot-v1
        string = f"python train_iq.py agent=softq env=acrobot eval.demos={eval_demos} eval.subsample_freq=5 method.chi=True method.loss=value_expert seed={seed} agent.critic_lr=1e-4"
        command = f"command{seed}Acrobotiqoffline{eval_demos}"
        command=f"{command}.txt"
        with open(command, 'w') as file:
            file.write(f'{string}\n')

        os.system(f'sbatch --job-name=IQL{seed} submit.sh {command}')
        #os.system(string)
        #CartPole-v1
        string = f"python train_iq.py agent=softq env=cartpole eval.demos={eval_demos} eval.subsample_freq=20 method.chi=True method.loss=value_expert agent.init_temperature=0.001 seed={seed} agent.critic_lr=1e-4"
       
        #os.system(string)

        #string = f"python train_iq.py agent=softq env=lunarlander eval.demos={eval_demos} eval.subsample_freq=5 method.chi=True method.loss=value_expert agent.init_temperature=0.001 seed={seed}"
        #os.system(string)

        command = f"command{seed}CartPoleiqoffline{eval_demos}"
        command=f"{command}.txt"
        with open(command, 'w') as file:
            file.write(f'{string}\n')
        
        os.system(f'sbatch --job-name=IQL{seed} submit.sh {command}')
