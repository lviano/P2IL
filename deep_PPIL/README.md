Deep Proximal Point Imitation Learning

To reproduce the Ant-v2 results run the command:

"""
train.py agent=sac env=ant expert.demos=10 method.loss=value_expert seed=0 agent.ppil=true train.use_target=false method.regularize=True agent.actor_lr=3e-05 agent.critic_lr=0.0003 agent.init_temp=0.01
"""


To reproduce the HalfCheetah-v2 results, run:


"""
train.py agent=sac env=cheetah expert.demos=10 method.tanh=false method.loss=value_expert seed=0 agent.ppil=true train.use_target=false method.regularize=True agent.actor_lr=3e-5 agent.critic_lr=0.0003 agent.tau=0.08 agent.online=false agent.init_temp=0.01
"""

To reproduce, the Atari Pong results, run:

"""
train.py agent=softq env=pong agent.init_temp=1e-3 method.loss=value_expert method.chi=True seed=0 expert.demos=40 agent.ppil=true agent.online=false agent.tau=8e-2
"""