import os
import argparse

#parser = argparse.ArgumentParser(description='Grid search hyperparameters')

#parser.add_argument('--env-name', metavar='G',
#                    help='name of the environment to run')
#args = parser.parse_args()
for env in ["cheetah","hopper","ant","walker"]:
    for loss in ["value_expert", "value", "v0"]:
        for actor_lr in [1e-5, 3e-5, 1e-4]:
            for critic_lr in [3e-3, 3e-2, 3e-4]:
                for alpha in [1e-2,1e-3,1e-4]:
                    for tau in [1e-1, 1, 1e1]:
                        for online in ["true", "false"]:
                            for seed in range(1):
                                string = f"python train_iq.py agent=sac env={env} \
                                    expert.demos=10 \
                                        method.loss={loss} seed={seed} \
                                            agent.ppil=true train.use_target=false\
                                                method.regularize=True \
                                                    agent.actor_lr={actor_lr} \
                                                        agent.critic_lr={critic_lr} \
                                                            agent.tau={tau} agent.online={online} \
                                                                agent.init_temp={alpha}"
                                command = f"command{seed}{env}{loss}{actor_lr}{critic_lr}{alpha}"
                                command=f"{command}.txt"
                                with open(command, 'w') as file:
                                    file.write(f'{string}\n')
                                
                                os.system(f'sbatch --job-name={env}{loss}{seed} submit.sh {command}')
