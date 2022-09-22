import wandb
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("viano/Acrobot-v1")
summary_list = [] 
config_list = [] 
name_list = [] 
dict_list = []
name_list_iq = [] 
dict_list_iq = []
for i, run in enumerate(runs):
    if run.name in ["fanciful-wind-774","treasured-snowball-773","vital-music-772","confused-sound-771","blooming-paper-770","sandy-star-769","azure-bee-768","silvery-music-767","astral-elevator-766","kind-terrain-765"]:
        time_list = []
        rewards = [] 
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files 
        #import pdb; pdb.set_trace()
        for step in run.scan_history():
            if "train/episode_reward" in step.keys():
                time_list.append(step["global_step"])
                rewards.append(step["train/episode_reward"])
        dict_list.append({"step":time_list, "rewards":rewards})

        # run.name is the name of the run.
        name_list.append(run.name)       

    if run.name in ["vocal-galaxy-778", "fanciful-tree-777", "exalted-galaxy-776","crisp-plasma-775"]:
        time_list = []
        rewards = [] 
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files 
        #import pdb; pdb.set_trace()
        for step in run.scan_history():
            if "train/episode_reward" in step.keys():
                time_list.append(step["global_step"])
                rewards.append(step["train/episode_reward"])
        dict_list_iq.append({"step":time_list, "rewards":rewards})

        # run.name is the name of the run.
        name_list_iq.append(run.name)
dictionary = {k:v for k,v in zip(name_list,dict_list)}
dictionary_iq = {k:v for k,v in zip(name_list_iq,dict_list_iq)}
import pickle

with open("data_acrobot.pkl", "wb") as f:
    pickle.dump(dictionary, f)
with open("data_acrobot_iq.pkl", "wb") as f:
    pickle.dump(dictionary_iq, f)