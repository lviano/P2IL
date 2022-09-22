import wandb
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("viano/CartPole-v1")
summary_list = [] 
config_list = [] 
name_list = [] 
dict_list = []
name_list_iq = [] 
dict_list_iq = []
for i, run in enumerate(runs):
    if run.name in ["solar-terrain-972", "sunny-thunder-971","dutiful-sun-970","upbeat-glitter-969","youthful-firebrand-968","glorious-universe-947","exalted-smoke-946","denim-salad-945","bumbling-energy-944"]:
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

    if run.name in ["likely-frog-943","denim-glade-942","dazzling-waterfall-941","ancient-moon-940","dandy-terrain-939"]:
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

with open("data_cartpole.pkl", "wb") as f:
    pickle.dump(dictionary, f)
with open("data_cartpole_iq.pkl", "wb") as f:
    pickle.dump(dictionary_iq, f)