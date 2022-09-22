**Instructions to reproduce the results of Proximal Point Imitation Learning**

**SingleChainProblem-v0**

```
python launch_many_seeds.py --env-name SingleChainProblem-v0 --expert-traj-path SingleChainProblem-v0_expert_traj.pt --learning-rate-w 0.3 --learning-rate-theta 0.005 --n-trajs 50 --optimizer adam
```

Set --T 300

**DoubleChainProblem-v0**

```
python launch_many_seeds.py --env-name DoubleChainProblem-v0 --expert-traj-path DoubleChainProblem-v0_expert_traj.pt --learning-rate-w 0.5 --learning-rate-theta 0.005 --n-trajs 50 --optimizer adam
```

Set --T 300

**WindyGrid-v0**

```
python launch_many_seeds.py --env-name WindyGrid-v0 --expert-traj-path WindyGrid-v0_expert_traj.pt --learning-rate-w 0.5 --learning-rate-theta 0.01 --n-trajs 50 --optimizer forb
```

Set --T 20

**RiverSwim-v0**

```
python launch_many_seeds.py --env-name RiverSwim-v0 --expert-traj-path RiverSwim-v0_expert_traj.pt --learning-rate-w 0.2 --learning-rate-theta 0.2 --n-trajs 50 --optimizer forb
```

Set --T 40


**WideTree-v0**

```
python launch_many_seeds.py --env-name WideTree-v0 --expert-traj-path WideTree-v0_expert_traj.pt --learning-rate-w 0.5 --learning-rate-theta 0.5 --n-trajs 25 --optimizer forb
```

Set --T 20

**TwoStateStochastic-v0**

```
 python launch_many_seeds.py --env-name TwoStateStochastic-v0 --expert-traj-path TwoStateStochastic-v0_expert_traj.pt --learning-rate-w 0.5 --learning-rate-theta 0.5 --n-trajs 25 --optimizer adam
```

Set --T 20

**TwoStateProblem-v0**

```
python launch_many_seeds.py --env-name TwoStateProblem-v0 --expert-traj-path TwoStateProblem-v0_expert_traj.pt --learning-rate-w 0.5 --learning-rate-theta 0.5 --n-trajs 25 --optimizer forb
```

Set --T 20
