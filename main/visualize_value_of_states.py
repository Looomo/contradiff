import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
os.chdir(parent_directory)
grandparent_directory = os.path.dirname(parent_directory)
sys.path.append(grandparent_directory)
import argparse
import umap
from scipy.spatial.distance import cdist
from matplotlib.colors import Normalize
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'normal'
import os
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 全局设置字体为 Times New Roman
rcParams['font.family'] = 'Times New Roman'

path_map = {
    'maze2d': '/home/$USER/sync/CDiffuser/maze2d_ug/dataset_infos',
    'kitchen': '/home/$USER/sync/CDiffuser/kitchen/dataset_infos',
    'gym': '/home/$USER/sync/CDiffuser/main/dataset_infos',
}

value_path_map = {
    'maze2d': '/home/$USER/sync/CDiffuser/maze2d_ug/dataset_infos',
    'kitchen': '/home/$USER/sync/CDiffuser/kitchen/dataset_infos',
    'mujoco': '/home/$USER/sync/CDiffuser/main/dataset_infos',
}





def determain_env(env_full_name):

    what_env_is_it = {
        'maze2d': 'maze2d' in env_full_name,
        'antmaze': 'antmaze-' in env_full_name,
        'adroit': 'pen-' in env_full_name or 'hammer-' in env_full_name or 'door-' in env_full_name or 'relocate-' in env_full_name,
        'gym': '-v2' in env_full_name,
        'kitchen': 'kitchen' in env_full_name
    }

    for k in what_env_is_it:
        if what_env_is_it[k]:
            return k

def sort_by_values(to_sort, values):
    inds = np.argsort(values)[::-1]
    if to_sort is not None:
        to_sort = to_sort[inds]
    values = values[inds]
    return to_sort, values


def visualize(name):

    save_path = "./intro/sub/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    which_env = determain_env(name)
    load_path = os.path.join(path_map[which_env], f"cluster_infos_{name}.pkl")
    value_load_path = os.path.join(path_map[which_env].replace("dataset_infos","value_of_states"), f"{name}.pkl")

    with open(load_path, "rb") as f:
        cluster_info = pkl.load(f)

    with open(value_load_path, "rb") as f:
        value_of_states = pkl.load(f)   
    states = cluster_info['mixed_states']
    values = cluster_info['mixed_vs']

    idxs, _ = sort_by_values( np.arange(len(values)), values  )

    if len(idxs) > 5000:
        idxs_to_vis = idxs[::len(idxs)//5000]
    else:
        idxs_to_vis = idxs

    states_to_vis = states[idxs_to_vis]
    vs_to_vis = values[idxs_to_vis]
    plt.clf()
    plt.cla()
    cmap = plt.cm.jet

    # norm = Normalize(vmin=min(vs_to_vis), vmax=max(vs_to_vis))
    # manhattan_distances = cdist(states_to_vis, states_to_vis, metric= "canberra" )#"euclidean")

    umap_reducer = umap.UMAP(n_components=2)
    embedding_all = umap_reducer.fit_transform(states_to_vis)


    plt.clf()
    plt.cla()
    plt.figure()
    data = embedding_all
    plt.scatter(data[:, 0], data[:, 1], cmap='viridis', s = 2, c = vs_to_vis)
    
    plt.xticks([]) 
    plt.yticks([])
    base_filename = f"{name}"
    plt.colorbar( label='Value of States')

    plt.subplots_adjust(wspace=0.04, bottom=0.01, top = 0.99, left=0.005, right=0.98)
    plt.savefig(os.path.join( save_path, f"{base_filename}.png" ), dpi = 600)
    
    return


import seaborn as sns

def pdf_of_acucmulated_rewards(name):
    plt.clf()
    plt.cla()
    plt.figure(figsize=(2.5,2.5))
    save_path = "./intro/pdf/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    which_env = determain_env(name)
    load_path = os.path.join(path_map[which_env], f"cluster_infos_{name}.pkl")
    value_load_path = os.path.join(path_map[which_env].replace("dataset_infos","value_of_states"), f"{name}.pkl")

    with open(load_path, "rb") as f:
        cluster_info = pkl.load(f)

    with open(value_load_path, "rb") as f:
        value_of_states = pkl.load(f)   
    states = cluster_info['mixed_states']
    values = cluster_info['mixed_vs']


    returns = [
        np.sum(traj['rewards']) for traj in cluster_info['mixed_trajectories']
    ]
    sns.kdeplot(returns,  fill=True, linewidth = 1.0, color="#e5541d")
    plt.xlim(0, np.max(returns))
    plt.xticks([0,10,20,30,40])
    plt.xlabel("Returns")
    plt.ylabel("Probability Density")    
    # plt.xticks([]) 
    # plt.yticks([])
    base_filename = f"{name}"
    plt.subplots_adjust(bottom=0.3, left=0.3,right=0.98, top = 0.98)
    plt.savefig(os.path.join( save_path, f"pdf_{base_filename}.png" ), dpi = 600)
    
    return


def acucmulated_rewards_v2(name):
    plt.clf()
    plt.cla()
    plt.figure(figsize=(8,8))
    save_path = "./intro/pdf/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    which_env = determain_env(name)
    load_path = os.path.join(path_map[which_env], f"cluster_infos_{name}.pkl")
    value_load_path = os.path.join(path_map[which_env].replace("dataset_infos","value_of_states"), f"{name}.pkl")

    with open(load_path, "rb") as f:
        cluster_info = pkl.load(f)

    with open(value_load_path, "rb") as f:
        value_of_states = pkl.load(f)   
    states = cluster_info['mixed_states']
    values = cluster_info['mixed_vs']

    returns = [
    ]
    trajs = cluster_info['mixed_trajectories']
    max_length = 600 # np.max( [len(_['rewards'])  for _ in trajs] )
    for traj in trajs:
        padding_length = max_length - len(traj['rewards'])
        padding = padding_length*traj['rewards'][-1] if len(traj['rewards']> 0) else 0
        returns.append(np.sum(traj['rewards']) + padding)

    idx, sorted_values = sort_by_values(  np.arange(len(returns)), np.array(returns)   )
    returns = sorted_values[::len(sorted_values)//100]
    returns = returns[-100:]
    plt.bar(len(returns), returns)
    # plt.xticks([0,10,20,30,40])
    plt.xlim(0,100)
    plt.xlabel("")
    plt.ylabel("Returns")    
    # plt.xticks([]) 
    # plt.yticks([])
    base_filename = f"{name}"
    plt.subplots_adjust(bottom=0.3, left=0.3,right=0.98, top = 0.98)
    plt.savefig(os.path.join( save_path, f"hel_{base_filename}.png" ), dpi = 600)
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="maze2d-umaze-v1", type=str)
    args = parser.parse_args()

    datasets_maze = [  f"maze2d-{data}-v1" for data in ['medium', 'large', 'umaze']    ]

    datasets_kitchen = [  f"kitchen-{data}-v0" for data in ['complete', 'partial', 'mixed']    ]

    mujoco = [
        f"{env}-{data}-v2" for env in ["walker2d", "hopper", "halfcheetah"] for data in  [ "medium", "medium-replay", "random" , "medium-expert"   ]
    ]

    # for name in datasets_kitchen + datasets_maze +  mujoco:
    #     # visualize(name)
    #     pdf_of_acucmulated_rewards(name)
    #     # break

    acucmulated_rewards_v2("maze2d-large-v1")