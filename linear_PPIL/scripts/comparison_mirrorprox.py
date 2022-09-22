import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.style.use('seaborn')

GAIL = {}
path="lpimi/algorithm/baselines/GAIL/assets/to_plot/"
for env in ["SingleChainProblem-v0", "RiverSwim-v0", "DoubleChainProblem-v0","WideTree-v0", "TwoStateProblem-v0","TwoStateStochastic-v0","WindyGrid-v0"]:
    GAIL[env]=[]
    for seed in range(50):
        to_plot = {"total_return":[],
                "total_counter":[]}
        with open(path+"/"+env+"_gail_"+str(seed)+".p", "rb") as f:
            data = pickle.load(f)
            for d in data:
                step, reward = d
                to_plot["total_return"].append(reward)
                to_plot["total_counter"].append(step)
        GAIL[env].append(to_plot)

LINEAR_GAIL = {}
path="lpimi/algorithm/baselines/GAIL/assets/to_plot/linear/"
for env in ["SingleChainProblem-v0", "RiverSwim-v0", "DoubleChainProblem-v0","WideTree-v0", "TwoStateProblem-v0","TwoStateStochastic-v0","WindyGrid-v0"]:
    LINEAR_GAIL[env]=[]
    for seed in range(50):
        to_plot = {"total_return":[],
                "total_counter":[]}
        with open(path+"/"+env+"_gail_"+str(seed)+".p", "rb") as f:
            data = pickle.load(f)
            for d in data:
                step, reward = d
                to_plot["total_return"].append(reward)
                to_plot["total_counter"].append(step)
        LINEAR_GAIL[env].append(to_plot)

AIRL = {}
path="lpimi/algorithm/baselines/GAIL/assets/to_plot/"
for env in ["SingleChainProblem-v0", "RiverSwim-v0", "DoubleChainProblem-v0","WideTree-v0", "TwoStateProblem-v0","TwoStateStochastic-v0","WindyGrid-v0"]:
    AIRL[env]=[]
    for seed in range(50):
        to_plot = {"total_return":[],
                "total_counter":[]}
        with open(path+"/"+env+"_airl_"+str(seed)+".p", "rb") as f:
            data = pickle.load(f)
            for d in data:
                step, reward = d
                to_plot["total_return"].append(reward)
                to_plot["total_counter"].append(step)
        AIRL[env].append(to_plot)

LINEAR_AIRL = {}
path="lpimi/algorithm/baselines/GAIL/assets/to_plot/linear/"
for env in ["SingleChainProblem-v0", "RiverSwim-v0", "DoubleChainProblem-v0","WideTree-v0", "TwoStateProblem-v0","TwoStateStochastic-v0","WindyGrid-v0"]:
    LINEAR_AIRL[env]=[]
    for seed in range(50):
        to_plot = {"total_return":[],
                   "total_counter":[]}
        with open(path+"/"+env+"_airl_"+str(seed)+".p", "rb") as f:
            data = pickle.load(f)
            for d in data:
                step, reward = d
                to_plot["total_return"].append(reward)
                to_plot["total_counter"].append(step)
        LINEAR_AIRL[env].append(to_plot)

dictionary = { "TwoStateProblem-v0": [lambda seed: "results/TwoStateProblem-v0"+str(seed)+"adamn_trajs5_lr_theta0.1_lr_w0.1.pt", #lambda seed: "results/TwoStateProblem"+str(seed)+"forb0.5.pt",
                                      #lambda seed: "results/TwoStateProblem"+str(seed)+"sgd0.1.pt",
                                      lambda seed: "results/PrimalDualTwoStateDeterministic"+str(seed)+"forb0.08.pt"],
                                      #lambda seed: "results/PrimalDualTwoStateDeterministic"+str(seed)+"sgd0.09_decay.pt"],
               "TwoStateStochastic-v0": [lambda seed: "results/TwoStateStochastic"+str(seed)+"forb0.5.pt",
                                         #lambda seed: "results/TwoStateStochastic"+str(seed)+"sgd.pt",
                                         lambda seed: "results/PrimalDualTwoStateStochastic"+str(seed)+"forb0.08.pt"],
                                         #lambda seed: "results/PrimalDualTwoStateStochastic"+str(seed)+"sgd0.08.pt"],
                "WideTree-v0": [lambda seed: "results/WideTree"+str(seed)+"forb0.5.pt",
                                lambda seed: "results/incrementalWideTree-v0"+str(seed)+"egn_trajs25_lr_theta0.5_lr_w0.5.pt"],
                                #lambda seed: "results/PrimalDualWideTree"+str(seed)+"forb0.5.pt"],
                                #lambda seed: "results/PrimalDualWideTree"+str(seed)+"_sgd.pt"],
                "RiverSwim-v0": [lambda seed: "results/RiverSwim-v0"+str(seed)+"adamn_trajs50_lr_theta0.005_lr_w0.5.pt", #lambda seed: "results/RiverSwim"+str(seed)+"forb0.2.pt",
                                 lambda seed: "results/incrementalRiverSwim-v0"+str(seed)+"egn_trajs25_lr_theta0.05_lr_w0.5.pt"],
                                #lambda seed: "results/PrimalDualRiverSwim"+str(seed)+"FORB.pt"],
                                #lambda seed: "results/PrimalDualRiverSwim"+str(seed)+"sgd0.1_350_trajs.pt"],
                "SingleChainProblem-v0": [lambda seed: "results/SingleChainProblem-v0"+str(seed)+"adamn_trajs50_lr_theta0.005_lr_w0.5.pt", #lambda seed: "results/SingleChain"+str(seed)+"forb0.1_n_trajs_50.pt",
                                          lambda seed: "results/incrementalSingleChainProblem-v0"+str(seed)+"egn_trajs25_lr_theta0.05_lr_w0.5.pt"],
                                          #lambda seed: "results/PrimalDualSingleChain"+str(seed)+"0.1.pt"],
                                          #lambda seed: "results/PrimalDualSingleChain"+str(seed)+"0.033.pt"],
                "DoubleChainProblem-v0": [lambda seed: "results/DoubleChainProblem-v0"+str(seed)+"adamn_trajs50_lr_theta0.005_lr_w0.5.pt", #lambda seed: "results/TrueDoubleChain"+str(seed)+".pt",
                                          lambda seed: "results/incrementalDoubleChainProblem-v0"+str(seed)+"egn_trajs25_lr_theta0.05_lr_w0.5.pt"],
                                          #lambda seed: "results/PrimalDualDoubleChain"+str(seed)+"forb0.025_n_dual_updates_5_ok.pt"],
                                          #lambda seed: "results/PrimalDualDoubleChain"+str(seed)+"sgd0.033_n_dual_updates_5.pt"],
                "WindyGrid-v0": [lambda seed: "results/WindyGrid-v0"+str(seed)+"forbn_trajs50_lr_theta0.01_lr_w0.5.pt",#lambda seed: "results/WindyGrid-v0"+str(seed)+"forbn_trajs100_lr0.005lr_w_0.5.pt",
                                 lambda seed: "results/incrementalWindyGrid-v0"+str(seed)+"egn_trajs50_lr_theta0.1_lr_w0.3.pt"]}##lambda seed: "results/WindyGrid-v0"+str(seed)+"sgdn_trajs200_lr_theta0.5_lr_w0.5.pt",
                                 #lambda seed: "results/primal_dualWindyGrid-v0"+str(seed)+"forbn_trajs50_lr_theta0.0006_lr_w0.5.pt"]}
                                 #lambda seed: "results/primal_dualWindyGrid-v0"+str(seed) +"sgdn_trajs200_lr_theta0.5_lr_w0.1.pt"]  }
seed_ranges= { "TwoStateProblem-v0": [range(10),np.arange(0,20,2)],
               "TwoStateStochastic-v0": [np.arange(0,20,2),np.arange(0,20,2)],
               "WideTree-v0": [np.arange(0,20,2), range(10)],
               "RiverSwim-v0": [range(10), range(10)],
               "SingleChainProblem-v0": [range(10), range(10)],
               "DoubleChainProblem-v0": [range(10), range(10)],
               "WindyGrid-v0": [range(50), range(50)]}
colors = ["blue","red","red","orange"]
limits = {"TwoStateProblem-v0": 20, 
          "TwoStateStochastic-v0": 30, 
          "WideTree-v0": 100, 
          "RiverSwim-v0": 175,
          "SingleChainProblem-v0": 150,
          "DoubleChainProblem-v0": 300,
          "WindyGrid-v0": 500}
IQLearn_best_params = { "TwoStateProblem-v0": [0.9, 1], 
                        "TwoStateStochastic-v0": [0.9, 1], 
                        "WideTree-v0": [0.9, 3], 
                        "RiverSwim-v0": [0.9, 1],
                        "SingleChainProblem-v0": [0.9, 1],
                        "DoubleChainProblem-v0": [0.9, 1],
                        "WindyGrid-v0": [0.9, 5]}
IQLearn_best_params_forb = { "TwoStateProblem-v0": [0.9, 1], 
                        "TwoStateStochastic-v0": [0.9, 1], 
                        "WideTree-v0": [0.9, 3], 
                        "RiverSwim-v0": [0.9, 3],
                        "SingleChainProblem-v0": [0.9, 1],
                        "DoubleChainProblem-v0": [0.9, 1],
                        "WindyGrid-v0": [0.9, 1]}
max_envs = [398.96, 198.76, 65.82, 156.04, 1949.4, 1947.78, 9438]
min_envs = [GAIL[env][0]["total_return"][0] for env in ["TwoStateProblem-v0", 
            "TwoStateStochastic-v0", 
            "WideTree-v0", 
            "RiverSwim-v0",
            "SingleChainProblem-v0",
            "DoubleChainProblem-v0",
            "WindyGrid-v0"]]
for env, max_env, min_env in zip(["TwoStateProblem-v0", 
            "TwoStateStochastic-v0", 
            "WideTree-v0", 
            "RiverSwim-v0",
            "SingleChainProblem-v0",
            "DoubleChainProblem-v0",
            "WindyGrid-v0"], max_envs, min_envs):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Plot our algorithms
    for files, seed_range, c in zip(dictionary[env],seed_ranges[env], colors):
        x_tot=None
        y_tot=None
        for seed in seed_range:
            with open(files(seed),"rb") as f:
                y, x = pickle.load(f)
                if x_tot is not None:
                    x_tot = np.vstack([x_tot, x])
                    y_tot = np.vstack([y_tot, y])
                elif x_tot is None:
                    x_tot = x
                    y_tot = y
        m = ( np.mean(y_tot, axis=0) - min_env)/(max_env - min_env)
        std = (np.std(y_tot, axis=0))/(max_env - min_env)
        ax.plot(np.mean(x_tot, axis=0)/100,m, "-o", color=c)
        ax.fill_between(np.mean(x_tot, axis=0)/100,
                        m + std,
                        m - std, 
                        alpha = 0.1,
                        facecolor=c)
    hor=np.arange(0,10000)
    ax.plot(hor,np.ones_like(hor), "--", color="gray")
    ax.xaxis.set_major_locator(MaxNLocator(5)) 
    ax.yaxis.set_major_locator(MaxNLocator(2)) 
    ax.set_yticks([0,1])
    ax.legend(["Proximal Point"," ", "MirrorProx"],fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel("Samples (x 100)",fontsize=30)
    #plt.ylabel("Normalized Total Return",fontsize=30)
    plt.xlim([-1,limits[env]])
    plt.ylim([-0.1, 1.1])
    plt.tight_layout()
    plt.savefig("figs/incremental"+env+"_normalized.pdf")