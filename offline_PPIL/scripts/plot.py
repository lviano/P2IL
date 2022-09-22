import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from os.path import exists
from matplotlib.ticker import MaxNLocator
parser = argparse.ArgumentParser(description='Plot')
parser.add_argument('--env-name', metavar='G',
                    help='name of the environment to run')
args = parser.parse_args()
index = 0
Y_max = 0
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for method,c in zip(["logistic", "logistic_primal_dual", "logistic_offline"], ["red","green", "blue", "purple"]):
    x_tot=None
    y_tot=None
    for seed in range(50):
        with open("pickle_results/"+method+"/"+args.env_name+str(seed)
                    +"n_trajs50000"
                    +"_lr_w0.005"
                    +"_lr_theta0.005.pt","rb") as f:
            
                        
            y , x = pickle.load(f)
            #ax.plot(x_tot, y_tot, color = c)
            if x_tot is not None:
                min_size = np.min([len(y), y_tot.shape[1]])
                x_tot = np.vstack([x_tot[:,:min_size], x[:min_size]])
                y_tot = np.vstack([y_tot[:,:min_size], y[:min_size]])
            elif x_tot is None:
                x_tot = np.array([x])
                y_tot = np.array([y])

    ax.plot(np.mean(x_tot, axis=0)/10000,np.mean(y_tot, axis=0), label=method)
    ax.fill_between(np.mean(x_tot, axis=0)/10000,
            np.mean(y_tot, axis=0) + np.std(y_tot, axis=0),
            np.mean(y_tot, axis=0) - np.std(y_tot, axis=0), alpha=0.1)
        
    ax.xaxis.set_major_locator(MaxNLocator(5)) 
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel("Samples (x10000)",fontsize=30)
    plt.ylabel("Total Return",fontsize=30)
    plt.legend()
    plt.tight_layout()
    file_name = "figs/"+args.env_name+".pdf"
    plt.savefig(file_name)