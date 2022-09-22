import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.style.use('seaborn')
dictionary = { "TwoStateProblem-v0": [lambda seed: "results/TwoStateProblem-v0"+str(seed)+"forbn_trajs25_lr_theta0.5_lr_w0.5.pt",#lambda seed: "results/TwoStateProblem"+str(seed)+"forb0.5.pt",
                                      lambda seed: "results/TwoStateProblem"+str(seed)+"sgd0.1.pt",
                                      lambda seed: "results/PrimalDualTwoStateDeterministic"+str(seed)+"forb0.08.pt",
                                      lambda seed: "results/PrimalDualTwoStateDeterministic"+str(seed)+"sgd0.09_decay.pt",
                                      lambda seed: "results/TwoStateProblem-v0"+str(seed)+"adamn_trajs5_lr_theta0.1_lr_w0.1.pt"],
               "TwoStateStochastic-v0": [lambda seed: "results/TwoStateStochastic"+str(seed)+"forb0.5.pt",
                                         lambda seed: "results/TwoStateStochastic"+str(seed)+"sgd.pt",
                                         lambda seed: "results/PrimalDualTwoStateStochastic"+str(seed)+"forb0.08.pt",
                                         lambda seed: "results/PrimalDualTwoStateStochastic"+str(seed)+"sgd0.08.pt",
                                         lambda seed: "results/TwoStateStochastic-v0"+str(seed)+"adamn_trajs25_lr_theta0.5_lr_w0.5.pt"],
                "WideTree-v0": [lambda seed: "results/WideTree-v0"+str(seed)+"forbn_trajs25_lr_theta0.5_lr_w0.5.pt",#lambda seed: "results/WideTree"+str(seed)+"forb0.5.pt",
                                lambda seed: "results/WideTree"+str(seed)+"sgd0.1.pt",
                                lambda seed: "results/PrimalDualWideTree"+str(seed)+"forb0.5.pt",
                                lambda seed: "results/PrimalDualWideTree"+str(seed)+"_sgd.pt",
                                lambda seed: "results/WideTree-v0"+str(seed)+"adamn_trajs25_lr_theta0.005_lr_w0.5.pt"],
                "RiverSwim-v0": [lambda seed: "results/RiverSwim"+str(seed)+"forb0.2_bis.pt",#lambda seed: "results/RiverSwim-v0"+str(seed)+"forbn_trajs25_lr_theta0.1_lr_w0.1.pt",#lambda seed: "results/RiverSwim"+str(seed)+"forb0.2.pt",
                                lambda seed: "results/RiverSwim"+str(seed)+"sgd0.6.pt",
                                lambda seed: "results/PrimalDualRiverSwim"+str(seed)+"FORB.pt",
                                lambda seed: "results/PrimalDualRiverSwim"+str(seed)+"sgd0.1_350_trajs.pt",
                                lambda seed: "results/RiverSwim-v0"+str(seed)+"adamn_trajs50_lr_theta0.005_lr_w0.5.pt"],
                "SingleChainProblem-v0": [lambda seed: "results/SingleChain"+str(seed)+"forb0.1_n_trajs_50.pt",
                                          lambda seed: "results/SingleChain"+str(seed)+"sgd0.1_n_trajs_50.pt",
                                          lambda seed: "results/PrimalDualSingleChain"+str(seed)+"0.1.pt",
                                          lambda seed: "results/PrimalDualSingleChain"+str(seed)+"0.033.pt",
                                          lambda seed: "results/SingleChainProblem-v0"+str(seed)+"adamn_trajs50_lr_theta0.005_lr_w0.3.pt"],
                "DoubleChainProblem-v0": [lambda seed: "results/TrueDoubleChain"+str(seed)+".pt",
                                          lambda seed: "results/TrueDoubleChain"+str(seed)+"sgd0.01.pt",
                                          lambda seed: "results/PrimalDualDoubleChain"+str(seed)+"forb0.025_n_dual_updates_5_ok.pt",
                                          lambda seed: "results/PrimalDualDoubleChain"+str(seed)+"sgd0.033_n_dual_updates_5.pt",
                                          lambda seed: "results/DoubleChainProblem-v0"+str(seed)+"adamn_trajs50_lr_theta0.005_lr_w0.5.pt"],
                "WindyGrid-v0": [lambda seed: "results/WindyGrid-v0"+str(seed)+"forbn_trajs50_lr_theta0.01_lr_w0.5.pt",#lambda seed: "results/WindyGrid-v0"+str(seed)+"forbn_trajs100_lr0.005lr_w_0.5.pt",
                                 lambda seed: "results/WindyGrid-v0"+str(seed)+"sgdn_trajs200_lr_theta0.5_lr_w0.5.pt",
                                 lambda seed: "results/primal_dualWindyGrid-v0"+str(seed)+"forbn_trajs50_lr_theta0.0006_lr_w0.5.pt",
                                 lambda seed: "results/primal_dualWindyGrid-v0"+str(seed) +"sgdn_trajs200_lr_theta0.5_lr_w0.1.pt",
                                 lambda seed: "results/WindyGrid-v0"+str(seed)+"adamn_trajs50_lr_theta0.005_lr_w0.5.pt"]  }
seed_ranges= { "TwoStateProblem-v0": [np.arange(0,20,2),np.arange(0,20,2),np.arange(0,20,2),np.arange(0,20,2), range(10)],
               "TwoStateStochastic-v0": [np.arange(0,20,2),np.arange(0,20,2),np.arange(0,20,2),np.arange(0,20,2),np.arange(0,20,2)],
               "WideTree-v0": [np.arange(0,20,2),np.arange(0,20,2),range(10), range(10), range(10)],
               "RiverSwim-v0": [np.arange(0,20,2),np.arange(0,20,2),range(10), range(10), range(10)],
               "SingleChainProblem-v0": [np.arange(0,20,2), np.arange(0,20,2),range(10), range(10), range(10)],
               "DoubleChainProblem-v0": [np.arange(0,20,2),np.arange(0,20,2),range(10), range(10), range(10)],
               "WindyGrid-v0": [range(50), range(50),range(50),range(50), range(10)]}
colors = ["blue","orange","green","red","turquoise"]
limits = {"TwoStateProblem-v0": 20, 
          "TwoStateStochastic-v0": 30, 
          "WideTree-v0": 100, 
          "RiverSwim-v0": 175,
          "SingleChainProblem-v0": 150,
          "DoubleChainProblem-v0": 120,
          "WindyGrid-v0": 1000}
for env in ["TwoStateProblem-v0", 
            "TwoStateStochastic-v0", 
            "WideTree-v0", 
            "RiverSwim-v0",
            "SingleChainProblem-v0",
            "DoubleChainProblem-v0",
            "WindyGrid-v0"]:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
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
        ax.plot(np.mean(x_tot, axis=0)/100,np.mean(y_tot, axis=0), "-o", color=c)
        ax.fill_between(np.mean(x_tot, axis=0)/100,
                        np.mean(y_tot, axis=0) + np.std(y_tot, axis=0),
                        np.mean(y_tot, axis=0) - np.std(y_tot, axis=0), 
                        alpha = 0.1,
                        facecolor=c)
    ax.xaxis.set_major_locator(MaxNLocator(5)) 
    ax.yaxis.set_major_locator(MaxNLocator(2)) 
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel("Samples (x 100)",fontsize=30)
    plt.ylabel("Total Return",fontsize=30)
    plt.xlim([-1,limits[env]])
    plt.tight_layout()
    plt.savefig("figs/"+env+"_ablation.pdf")