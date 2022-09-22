import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.style.use('seaborn')

GAIL = {}
path="lpimi/algorithm/baselines/GAIL/assets/to_plot/"
for env in ["Ant-v2"]:
    GAIL[env]=[]
    for seed in range(3):
        to_plot = {"total_return":[],
                "total_counter":[]}
        with open(path+"/"+env+"_gail_"+str(seed)+".p", "rb") as f:
            data = pickle.load(f)
            for d in data:
                step, reward = d
                to_plot["total_return"].append(reward)
                to_plot["total_counter"].append(step)
        GAIL[env].append(to_plot)

AIRL = {}
path="lpimi/algorithm/baselines/GAIL/assets/to_plot/"
for env in ["Ant-v2"]:
    AIRL[env]=[]
    for seed in range(3):
        to_plot = {"total_return":[],
                "total_counter":[]}
        with open(path+"/"+env+"_airl_"+str(seed)+".p", "rb") as f:
            data = pickle.load(f)
            for d in data:
                step, reward = d
                to_plot["total_return"].append(reward)
                to_plot["total_counter"].append(step)
        AIRL[env].append(to_plot)


for env, max_env, min_env in zip(["Ant-v2"], [2859.0], [900.0]):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    for algo, algo_label, algo_color in zip([GAIL, AIRL],
        ["GAIL", "AIRL"], ["gray",  "black"]):
        xs=None
        mean=None
        for seed_n in range(3):
            if xs is not None:
                xs = np.vstack([xs, algo[env][seed_n]["total_counter"]])
                mean = np.vstack([mean, algo[env][seed_n]["total_return"]])
            else:
                xs=algo[env][seed_n]["total_counter"]
                mean=algo[env][seed_n]["total_return"]
        ms = (np.mean(mean, axis=0) - min_env)/(max_env - min_env)
        std = (np.std(mean, axis=0))/(max_env - min_env)
        steps=3
        ax.plot(np.mean(xs,axis=0)[::steps]/100, ms[::steps],"-o", color=algo_color,label=algo_label)
        ax.fill_between(np.mean(xs,axis=0)[::steps]/100, ms[::steps]+std[::steps],ms[::steps]-std[::steps],
                    facecolor = algo_color, alpha=0.1)
    
    with open("results/data_ant_ours.pkl","rb") as f:
        dictionaries = pickle.load(f)
    
    xs=None
    mean=None
    for j, k in enumerate(dictionaries.keys()):
        d = dictionaries[k]
        if xs is None:
            mean = np.array([d["rewards"]]).reshape(1,-1)
            xs = np.array([d["step"]]).reshape(1,-1)
        else:
            to_add = np.array([d["rewards"]]).reshape(1,-1)
            s = to_add.shape[1]
            s = min(s,mean.shape[1])
            xs = np.vstack([xs[:,:s], np.array([d["step"]]).reshape(1,-1)[0,:s]])
            mean = np.vstack([mean[:,:s], to_add[0,:s]])

    ms = (np.mean(mean, axis=0) - min_env)/(max_env - min_env)
    std = (np.std(mean, axis=0))/(max_env - min_env)
    steps=3
    ax.plot(np.mean(xs,axis=0)[::steps]/100, ms[::steps],"-o", color="blue",label="Proximal Point")
    
    ax.fill_between(np.mean(xs,axis=0)[::steps]/100, ms[::steps]+std[::steps],ms[::steps]-std[::steps],
                facecolor = "blue", alpha=0.1)

    with open("results/data_ant_iq_ours.pkl","rb") as f:
        dictionaries = pickle.load(f)
    xs=None
    mean=None
    for d in dictionaries.values():
        
        if xs is None:
            mean = np.array([d["rewards"]]).reshape(1,-1)
            xs = np.array([d["step"]]).reshape(1,-1)
        else:
            to_add = np.array([d["rewards"]]).reshape(1,-1)
            s = to_add.shape[1]
            s = min(s,mean.shape[1])
            xs = np.vstack([xs[:,:s], np.array([d["step"]]).reshape(1,-1)[0,:s]])
            mean = np.vstack([mean[:,:s], to_add[0,:s]])

    ms = (np.mean(mean, axis=0) - min_env)/(max_env - min_env)
    std = (np.std(mean, axis=0))/(max_env - min_env)
    ax.plot(np.mean(xs,axis=0)[::3]/100, ms[::3],"-o", color="goldenrod",label="IQ")
    
    ax.fill_between(np.mean(xs,axis=0)[::3]/100, ms[::3]+std[::3],ms[::3]-std[::3],
                facecolor = "goldenrod", alpha=0.1)
    hor=np.arange(0,10000)
    ax.plot(hor,np.ones_like(hor), "--", color="gray")
    ax.xaxis.set_major_locator(MaxNLocator(5)) 
    ax.yaxis.set_major_locator(MaxNLocator(2)) 
    ax.set_yticks([0, 1.0])
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel("Samples (x 100)",fontsize=30)
    #plt.ylabel("Normalized Total Return",fontsize=30)
    plt.xlim([-10,2000])
    plt.ylim([-0.1, 1.4])
    plt.tight_layout()
    plt.savefig("figs/final/"+env+"_normalized.pdf")