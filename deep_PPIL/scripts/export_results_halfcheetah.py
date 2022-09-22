import wandb
import pickle

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("viano/HalfCheetah-v2") #runs = api.runs("imirl/HalfCheetah-v2")
"""summary_list = [] 
config_list = [] 
name_list = [] 
dict_list = []
for i, run in enumerate(runs):
    if run.name in ["serene-valley-620", "efficient-totem-619","confused-firefly-618", "balmy-silence-617", "northern-cloud-616"]: #["HalfCheetah-v2b956845c-ec90","HalfCheetah-v229bce234-a609","HalfCheetah-v25301065e-de94"]:
        time_list = []
        rewards = [] 
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files 
        #import pdb; pdb.set_trace()
        for step in run.scan_history():
            if "eval/episode_reward" in step.keys():
                time_list.append(step["global_step"])
                rewards.append(step["eval/episode_reward"])
        dict_list.append({"step":time_list, "rewards":rewards})

        # run.name is the name of the run.
        name_list.append(run.name) 
dictionary = {k:v for k,v in zip(name_list,dict_list)}
with open("data_halfcheetah.pkl", "wb") as f:
    pickle.dump(dictionary, f)
"""
name_list_iq = [] 
dict_list_iq = []
#runs = api.runs("viano/HalfCheetah-v2")
for i, run in enumerate(runs):
    if run.name in ["IQLearn0", "IQLearn1", "IQLearn3","IQLearn4"]:
        time_list = []
        rewards = [] 
        # run.summary are the output key/values like accuracy.
        # We call ._json_dict to omit large files 
        #import pdb; pdb.set_trace()
        for step in run.scan_history():
            if "eval/episode_reward" in step.keys():
                time_list.append(step["global_step"])
                rewards.append(step["eval/episode_reward"])
        dict_list_iq.append({"step":time_list, "rewards":rewards})

        # run.name is the name of the run.
        name_list_iq.append(run.name)

dictionary_iq = {k:v for k,v in zip(name_list_iq,dict_list_iq)}



with open("data_halfcheetah_iq.pkl", "wb") as f:
    pickle.dump(dictionary_iq, f)