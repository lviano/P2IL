## Requirement

- pytorch (>= 1.4)
- gym
- wandb
- tensorboardX
- hydra-core

## Installation

- Make a conda environment and install dependencies: `pip install -r requirements.txt`
- Setup wandb project to log and visualize metrics

Code based on [**(https://div99.github.io/IQ-Learn)**]


To reproduce the results of Our Algorithm on CartPole run:

```
python train.py agent=softq env=cartpole eval.demos=10 eval.subsample_freq=5 method.loss=value seed=0 method.type=logistic_offline agent.init_temperature=1
```

on Acrobot:

```
python train.py agent=softq env=acrobot eval.demos=10 eval.subsample_freq=5 method.loss=value seed=0 method.type=logistic_offline agent.init_temperature=1
```

on LunarLander:

```
python train.py agent=softq env=lunarlander eval.demos=10 eval.subsample_freq=5 method.loss=value seed=0 method.type=logistic_offline agent.init_temperature=0.001 agent.critic_lr=1e-4
```


For our results we run 10 seeds from (0 to 9)

