from statistics import mode
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import pylab as pl

seed_dict = [0,1,2,3,4,5,6,7,8,9,10]
need_4lif = True
length = 40000

# 4-LIF topologies: load from record/4-LIF/
_4LIF = ["LIF_1_3", "LIF_2_2", "LIF_1_2_1", "LIF_1_1_1_1", "LIF_ring"]
# 5-LIF topologies: load from record/5-LIF/
_5LIF = ["LIF_1_3_1", "LIF_1_1_3", "LIF_1_2_2", "LIF_1_1_2_1", "LIF_1_1_1_1_1", "LIF_1_4"]
# Single-dir topologies: load from record/{model_name}/
_OTHER = ["LIF", "HH", "LIF_HH", "4LIF", "ANN"]

ALL_MODELS = _OTHER + _4LIF + _5LIF

def record_subdir(model_name):
    if model_name in _4LIF:
        return "4-LIF"
    if model_name in _5LIF:
        return "5-LIF"
    return model_name

total_reward_dict = {m: [] for m in ALL_MODELS}
mean_reward_dict = {m: [] for m in ALL_MODELS}
std_reward_dict = {m: [] for m in ALL_MODELS}
label_dict = {
    "LIF": "s-LIF", "HH": "HH", "LIF_HH": "s-LIF2HH",
    "4LIF": "4s-LIF", "ANN": "ANN",
    "LIF_1_3": "1+3 LIF", "LIF_2_2": "2+2 LIF",
    "LIF_1_2_1": "1+2+1 LIF", "LIF_1_1_1_1": "1+1+1+1 LIF",
    "LIF_ring": "LIF ring",
    "LIF_1_3_1": "1+3+1 LIF", "LIF_1_1_3": "1+1+3 LIF",
    "LIF_1_2_2": "1+2+2 LIF", "LIF_1_1_2_1": "1+1+2+1 LIF",
    "LIF_1_1_1_1_1": "1+1+1+1+1 LIF", "LIF_1_4": "1+4 LIF",
}
for seed in seed_dict:
    for model_name in ALL_MODELS:
        if model_name == "4LIF" and not need_4lif:
            continue
        subdir = record_subdir(model_name)
        # File naming convention changed: {model_name}_seed{seed} inside grouped subdirs
        fname = "./record/{}/reward_iteration_{}_seed{}.npy".format(
            subdir, model_name, seed)
        # Fall back to old naming for the _OTHER models that still use the
        # old per-model subdir format
        if model_name in _OTHER:
            fname = "./record/{}/reward_iteration_{}.npy".format(
                subdir, model_name, seed)
        data = np.load(fname, allow_pickle=True)
        data = data.tolist()
        iteration = data["iteration"]
        reward_dict = data["reward_dict"]
        avg_reward_dict = reward_dict
        n = 1000
        for i in range(iteration - n):
            avg_reward_dict[i] = np.mean(reward_dict[i:i+n])
        total_reward_dict[model_name].append(avg_reward_dict[0:length])
        #plt.plot(iter, avg_reward_dict[0:iteration-n])

for model_name in ALL_MODELS:
    if(model_name == "4LIF" and not need_4lif):
        continue
    std_reward_dict[model_name] = np.std(total_reward_dict[model_name], axis=0)
    mean_reward_dict[model_name] = np.mean(total_reward_dict[model_name], axis=0)

iter = np.linspace(start=1, stop=length, num=length)
color_number = 0
color_dict = [[30,30,230],[220,180,30],[255,20,0],[150,80,200],[100,200,255],[255,100,200],[0,180,180],[180,100,255],[80,180,100],[255,150,80],[100,150,200],[200,100,150],[50,200,100],[150,50,200],[100,200,50]]
for model_name in ALL_MODELS:
    if(model_name == "4LIF" and not need_4lif):
        continue
    r,g,b = np.array(color_dict[color_number % len(color_dict)])/255
    color_number += 1
    plt.fill_between(iter, mean_reward_dict[model_name]-1*std_reward_dict[model_name],
                     mean_reward_dict[model_name]+1*std_reward_dict[model_name],
                     color=(r, g, b, 0.1))
    plt.plot(iter, mean_reward_dict[model_name], label=label_dict[model_name], linewidth=0.8, color = (r,g,b))
plt.legend(prop = {'size':17}, loc = "lower right")
plt.xlabel("iteration",fontsize = 15)
plt.ylabel("average reward",fontsize = 15)
plt.tick_params(labelsize=13) #调整坐标轴数字大小

ax = pl.gca()
ax.xaxis.get_major_formatter().set_powerlimits((0,1))

plt.savefig("reward_picture.png")