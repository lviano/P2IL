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

dictionary = { "TwoStateProblem-v0": [lambda seed: "results/TwoStateProblem-v0"+str(seed)+"forbn_trajs25_lr_theta0.5_lr_w0.5.pt"], #lambda seed: "results/TwoStateProblem"+str(seed)+"forb0.5.pt",
                                      #lambda seed: "results/TwoStateProblem"+str(seed)+"sgd0.1.pt",
                                      #lambda seed: "results/PrimalDualTwoStateDeterministic"+str(seed)+"forb0.08.pt"],
                                      #lambda seed: "results/PrimalDualTwoStateDeterministic"+str(seed)+"sgd0.09_decay.pt"],
               "TwoStateStochastic-v0": [lambda seed: "results/TwoStateStochastic-v0"+str(seed)+"adamn_trajs25_lr_theta0.5_lr_w0.5.pt"],
                                         #lambda seed: "results/TwoStateStochastic"+str(seed)+"sgd.pt",
                                         #lambda seed: "results/PrimalDualTwoStateStochastic"+str(seed)+"forb0.08.pt"],
                                         #lambda seed: "results/PrimalDualTwoStateStochastic"+str(seed)+"sgd0.08.pt"],
                "WideTree-v0": [lambda seed: "results/WideTree-v0"+str(seed)+"forbn_trajs25_lr_theta0.5_lr_w0.5.pt"],
                                #lambda seed: "results/WideTree"+str(seed)+"sgd0.1.pt",
                                #lambda seed: "results/PrimalDualWideTree"+str(seed)+"forb0.5.pt"],
                                #lambda seed: "results/PrimalDualWideTree"+str(seed)+"_sgd.pt"],
                "RiverSwim-v0": [lambda seed: "results/RiverSwim"+str(seed)+"forb0.2_bis.pt"], #lambda seed: "results/RiverSwim"+str(seed)+"forb0.2.pt",
                                #lambda seed: "results/RiverSwim"+str(seed)+"sgd0.6.pt",
                                #lambda seed: "results/PrimalDualRiverSwim"+str(seed)+"FORB.pt"],
                                #lambda seed: "results/PrimalDualRiverSwim"+str(seed)+"sgd0.1_350_trajs.pt"],
                "SingleChainProblem-v0": [lambda seed: "results/SingleChainProblem-v0"+str(seed)+"adamn_trajs50_lr_theta0.005_lr_w0.3.pt"], #lambda seed: "results/SingleChain"+str(seed)+"forb0.1_n_trajs_50.pt",
                                          #lambda seed: "results/SingleChain"+str(seed)+"sgd0.1_n_trajs_50.pt",
                                          #lambda seed: "results/PrimalDualSingleChain"+str(seed)+"0.1.pt"],
                                          #lambda seed: "results/PrimalDualSingleChain"+str(seed)+"0.033.pt"],
                "DoubleChainProblem-v0": [lambda seed: "results/DoubleChainProblem-v0"+str(seed)+"adamn_trajs50_lr_theta0.005_lr_w0.5.pt"], #lambda seed: "results/TrueDoubleChain"+str(seed)+".pt",
                                          #lambda seed: "results/TrueDoubleChain"+str(seed)+"sgd0.01.pt",
                                          #lambda seed: "results/PrimalDualDoubleChain"+str(seed)+"forb0.025_n_dual_updates_5_ok.pt"],
                                          #lambda seed: "results/PrimalDualDoubleChain"+str(seed)+"sgd0.033_n_dual_updates_5.pt"],
                "WindyGrid-v0": [lambda seed: "results/WindyGrid-v0"+str(seed)+"forbn_trajs50_lr_theta0.01_lr_w0.5.pt"],}#lambda seed: "results/WindyGrid-v0"+str(seed)+"forbn_trajs100_lr0.005lr_w_0.5.pt",
                                 #lambda seed: "results/WindyGrid-v0"+str(seed)+"sgdn_trajs200_lr_theta0.5_lr_w0.5.pt",
                                 #lambda seed: "results/primal_dualWindyGrid-v0"+str(seed)+"forbn_trajs50_lr_theta0.0006_lr_w0.5.pt"]}
                                 #lambda seed: "results/primal_dualWindyGrid-v0"+str(seed) +"sgdn_trajs200_lr_theta0.5_lr_w0.1.pt"]  }
seed_ranges= { "TwoStateProblem-v0": [range(50)],
               "TwoStateStochastic-v0": [range(50)],
               "WideTree-v0": [range(50)],
               "RiverSwim-v0": [range(50)],
               "SingleChainProblem-v0": [range(50)],
               "DoubleChainProblem-v0": [range(50)],
               "WindyGrid-v0": [range(50)]}
colors = ["blue","green","red","orange"]
limits = {"TwoStateProblem-v0": 20, 
          "TwoStateStochastic-v0": 30, 
          "WideTree-v0": 100, 
          "RiverSwim-v0": 50,
          "SingleChainProblem-v0": 150,
          "DoubleChainProblem-v0": 120,
          "WindyGrid-v0": 250}
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
    for algo, algo_label, algo_color in zip([GAIL, LINEAR_GAIL, AIRL, LINEAR_AIRL],
        ["GAIL", "GAIL_LINEAR", "AIRL", "AIRL_LINEAR"], ["gray", "darkcyan", "black", "brown"]):
        xs=None
        mean=None
        for seed_n in range(50):
            if xs is not None:
                xs = np.vstack([xs, algo[env][seed_n]["total_counter"]])
                mean = np.vstack([mean, algo[env][seed_n]["total_return"]])
            else:
                xs=algo[env][seed_n]["total_counter"]
                mean=algo[env][seed_n]["total_return"]
        ms = (np.mean(mean, axis=0) - min_env)/(max_env - min_env)
        std = (np.std(mean, axis=0))/(max_env - min_env)
        ax.plot(np.mean(xs,axis=0)/100, ms,"-o", color=algo_color,label=algo_label)
        ax.fill_between(np.mean(xs,axis=0)/100, ms+std,ms-std,
                    facecolor = algo_color, alpha=0.1)
    """
    # Plot IQLearn sgd
    lr_theta, n_trajs = IQLearn_best_params[env]
    x_tot=None
    y_tot=None
    max_seed = 50 # if env=="WideTree-v0" else 50
    for seed in range(max_seed):
        file_name = "results/iq_learn/"+env+str(seed) \
                +"sgdn_trajs"+str(n_trajs) \
                +"_lr_theta"+str(lr_theta)+".pt"
        with open(file_name,"rb") as f:
            y, x = pickle.load(f)
            if x_tot is not None:
                x_tot = np.vstack([x_tot, x])
                y_tot = np.vstack([y_tot, y])
            elif x_tot is None:
                x_tot = x
                y_tot = y
    if env in ["SingleChainProblem-v0", 
                "DoubleChainProblem-v0", "RiverSwim-v0"]:
        m = (np.mean(y_tot, axis=0)[:-1:3] -min_env)/(max_env - min_env)
        std = (np.std(y_tot, axis=0)[:-1:3])/(max_env - min_env)
        ax.plot(np.mean(x_tot, axis=0)[:-1:3]/100,m,"-o",color="yellow",label="IQ SGD")
        ax.fill_between(np.mean(x_tot, axis=0)[:-1:3]/100,
                m + std,
                m - std, alpha=0.1, facecolor="yellow")
         
    else:
        m = (np.mean(y_tot, axis=0)  -min_env)/(max_env - min_env)
        std = (np.std(y_tot, axis=0))/(max_env - min_env)
        ax.plot(np.mean(x_tot, axis=0)/100,m,"-o",color="yellow",label="IQ SGD")
        ax.fill_between(np.mean(x_tot, axis=0)/100,
                m + std,
                m - std, alpha=0.1, facecolor="yellow")
    """
    # Plot IQLearn forb
    lr_theta, n_trajs = IQLearn_best_params_forb[env]
    x_tot=None
    y_tot=None
    max_seed = 50 #if env=="WideTree-v0" else 50
    for seed in range(max_seed):
        file_name = "results/iq_learn/"+env+str(seed) \
                +"forbn_trajs"+str(n_trajs) \
                +"_lr_theta"+str(lr_theta)+".pt"
        with open(file_name,"rb") as f:
            y, x = pickle.load(f)
            if x_tot is not None:
                x_tot = np.vstack([x_tot, x])
                y_tot = np.vstack([y_tot, y])
            elif x_tot is None:
                x_tot = x
                y_tot = y
    if env in ["WindyGrid-v0",
                "SingleChainProblem-v0", 
                "DoubleChainProblem-v0"]:
        m = (np.mean(y_tot, axis=0)[:-1:3] -min_env)/(max_env - min_env)
        std = (np.std(y_tot, axis=0)[:-1:3])/(max_env - min_env)
        ax.plot(np.mean(x_tot, axis=0)[:-1:3]/100,m,"-o",color="goldenrod",label="IQ FORB")
        ax.fill_between(np.mean(x_tot, axis=0)[:-1:3]/100,
                m + std,
                m - std, alpha=0.1, facecolor="goldenrod")
         
    else:
        m = (np.mean(y_tot, axis=0)  -min_env)/(max_env - min_env)
        std = (np.std(y_tot, axis=0))/(max_env - min_env)
        ax.plot(np.mean(x_tot, axis=0)/100,m,"-o",color="goldenrod",label="IQ FORB")
        ax.fill_between(np.mean(x_tot, axis=0)/100,
                m + std,
                m - std, alpha=0.1, facecolor="goldenrod")

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
    ax.set_yticks([0, 1.0])
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel("Samples (x 100)",fontsize=30)
    #plt.ylabel("Normalized Total Return",fontsize=30)
    plt.xlim([-1,limits[env]])
    plt.ylim([-0.1, 1.1])
    plt.tight_layout()
    plt.savefig("figs/final/"+env+"_normalized.pdf")
