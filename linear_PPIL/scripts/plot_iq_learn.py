import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
parser = argparse.ArgumentParser(description='Plot IQ_LEARN')
parser.add_argument('--env-name', metavar='G',
                    help='name of the environment to run')
parser.add_argument('--optimizer', metavar='G',
                    help='optimizer')
args = parser.parse_args()
index = 0
Y_max = 0
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
for lr_theta in [5e-3, 1e-3, 5e-1, 1e-1, 5e-2, 1e-2, 9e-1]:
        for n_trajs in [1, 3, 5, 10]:
            x_tot=None
            y_tot=None
            for seed in range(10):
                file_name = "results/iq_learn/"+args.env_name+str(seed) \
                        +args.optimizer+"n_trajs"+str(n_trajs) \
                        +"_lr_theta"+str(lr_theta)+".pt"
                
                with open(file_name,"rb") as f:
                    y, x = pickle.load(f)
                    if x_tot is not None:
                        x_tot = np.vstack([x_tot, x])
                        y_tot = np.vstack([y_tot, y])
                    elif x_tot is None:
                        x_tot = x
                        y_tot = y
            ax.plot(np.mean(x_tot, axis=0)/10000,np.mean(y_tot, axis=0))
            ax.fill_between(np.mean(x_tot, axis=0)/10000,
                    np.mean(y_tot, axis=0) + np.std(y_tot, axis=0),
                    np.mean(y_tot, axis=0) - np.std(y_tot, axis=0), alpha=0.1)
            if np.mean(y_tot, axis=0)[10] > Y_max:
                Y_max = np.mean(y_tot, axis=0)[10]
                print(lr_theta, n_trajs)
ax.xaxis.set_major_locator(MaxNLocator(5)) 
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("Samples (x10000)",fontsize=30)
plt.ylabel("Total Return",fontsize=30)

plt.tight_layout()
file_name = "figs/IQLearn"+args.env_name+str(seed) \
                                        +args.optimizer+".pdf"
plt.savefig(file_name)