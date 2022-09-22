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

plt.style.use('seaborn')

index = 0
Y_max = 0
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
expert_values = {"CartPole-v1": 500, "Acrobot-v1": -63.678, "LunarLander-v2": 318.7}
min_values = {"CartPole-v1": 25, "Acrobot-v1": -500, "LunarLander-v2": -200}
for method,c in zip([ "iq_offline","logistic_offline"], [ "goldenrod","blue"]):
    x_tot = [1,3,7,10,15]
    y_tot = []
    y_std = []
    for n_trajs in x_tot:
        ys_max=[]
        for seed in range(10):
            if method=="logistic_offline":
                if args.env_name in ["CartPole-v1", "Acrobot-v1"]:
                    with open("pickle_results/"+method+"/"+args.env_name+str(seed)
                                +"n_trajs"+str(n_trajs)
                                +"_lr_w0.005"
                                +"_lr_theta0.005.pt","rb") as f:
                        _, y, _ = pickle.load(f)
                        #ax.plot(x_tot, y_tot, color = c)
                        ys_max.append(np.max(y))
                else:
                    if exists("pickle_results/"+method+"/init_temp_0.01_lunar_lander/"+args.env_name+str(seed)
                                +"n_trajs"+str(n_trajs)
                                +"_lr_w0.0001" #+"_lr_w0.005"
                                +"_lr_theta0.0001.pt"):
                        with open("pickle_results/"+method+"/init_temp_0.01_lunar_lander/"+args.env_name+str(seed)
                                    +"n_trajs"+str(n_trajs)
                                    +"_lr_w0.0001" #+"_lr_w0.005"
                                    +"_lr_theta0.0001.pt","rb") as f: #+"_lr_theta0.005.pt","rb") as f:
                            _, y, _ = pickle.load(f)
                            #ax.plot(x_tot, y_tot, color = c)
                            ys_max.append(np.max(y))
            else:
                if exists("pickle_results/"+method+"/"+args.env_name+str(seed)
                            +"n_trajs"+str(n_trajs)
                            +".pt"):
                    with open("pickle_results/"+method+"/"+args.env_name+str(seed)
                                +"n_trajs"+str(n_trajs)
                                +".pt","rb") as f:
            
                        _, y, _ = pickle.load(f)
                        #ax.plot(x_tot, y_tot, color = c)
                        ys_max.append(np.max(y))
        average_max=np.mean(ys_max)
        std = np.std(ys_max) 
        y_tot.append(average_max)
        y_std.append(std)
    
    label = "Ours Offline" if method == "logistic_offline" else "IQLearn"
    ax.plot(x_tot,y_tot, "-*", color=c, markersize=15, label=label)
    ax.fill_between(x_tot,
            np.array(y_tot) + np.array(y_std),
            np.array(y_tot) - np.array(y_std), alpha=0.1, facecolor=c)
        
ax.xaxis.set_major_locator(MaxNLocator(6)) 
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel("Expert Trajectories",fontsize=30)
plt.ylabel("Total Return",fontsize=30)
plt.legend(fontsize=30)
plt.tight_layout()
file_name = "figs/"+args.env_name+"_offline.pdf"
plt.savefig(file_name)