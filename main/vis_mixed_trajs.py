import sys
import os
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
os.chdir(parent_directory)
grandparent_directory = os.path.dirname(parent_directory)
sys.path.append(grandparent_directory)
import os
import re
import glob
from config.locomotion_config import Configs
import numpy as np
import torch
import shutil
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pickle as pkl
import argparse
import umap
import copy
from sklearn.manifold import TSNE
from matplotlib.colors import Normalize
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
import inspect
from operator import itemgetter
from config.locomotion_config import Configs
import diffuser.utils as utils
from global_rendering import MuJoCoRenderer
from dm_control.mujoco import engine
plt.rcParams['font.family'] = 'DeJavu Serif'
import seaborn as sns
matplotlib.rcParams['font.serif'] = 'Times New Roman'

from myutils import sort_by_values
fs = 39
def visual_values(env):

    ratios = [0.1,0.2,0.3]
    datasets = ["medium", "medium-replay", "random"]
    name = {
        'medium': "Halfcheetah-Medium",
        'medium-replay': "Halfcheetah-Med-Replay",
        'random': "Halfcheetah-Random",
    }


    fig,axes = plt.subplots(nrows = 1, ncols = 9, figsize = (25, 3),  sharex=True, sharey=True)
    axes[0].set_ylabel('Returns', fontsize = fs)
    for idx, ax in enumerate(axes):
        dataset = datasets[idx//3]

        ratio = ratios[idx%3]

        with open(f"dataset_infos/cluster_infos_{env}-{dataset}-{ratio:.2f}-v2.pkl", "rb") as f:
            infos = pkl.load(f)
        trajs = infos['mixed_trajectories']
        returns = [   np.sum(traj['rewards'])  for traj in trajs   ]
        idx_sorted, returns_sorted = sort_by_values(  np.arange(len(returns)), np.array(returns)  )


        returns_to_vis = list(returns_sorted[::len(returns)//100])
        padding_length = 100 - len(returns_to_vis)
        if padding_length > 0:
            for i in range(padding_length):
                returns_to_vis.append(returns_to_vis[-1])
        elif padding_length < 0:
            returns_to_vis = returns_to_vis[:padding_length]
        else:
            print("No need to padding")

        ax.bar(range(len(returns_to_vis)), returns_to_vis)
        if idx % 3 == 1:
            ax.set_title(f'{name[dataset]}-0.1, 0.2, 0.3', fontsize = fs)    
        # if idx % 3 == 2:
        #     ax.text(0.5, 0.01,f"Halfcheetah-{name[dataset]}", ha='center', va='bottom')
    plt.tight_layout(pad = 0.05)
    plt.subplots_adjust(bottom=0.1, left = 0.04)
    # x_min, x_max = plt.gca().get_xlim() 
    # positions = [ax.get_position() for ax in axes]
    # for idx in [2,5,7]:
    #     pos = positions[idx]
    #     x = (pos.x0 + pos.x1) / 2
    #     y = pos.y0 - 0.25  # 在子图下方适当位置添加文字
    #     fig.text(x, y, f"Halfcheetah-{name[dataset]}", ha='center', va='top')
    # plt.savefig("my.png", dpi = 600)
    plt.savefig("values_miaxed_halfcheetah.pdf", format = 'pdf')
    return
from scipy.stats import gaussian_kde

locations = {
    
}

def visual_values_pdf(env):

    ratios = [0.1,0.2,0.3]
    datasets = ["medium", "medium-replay", "random"]
    name = {
        'medium': "Halfcheetah-Medium",
        'medium-replay': "Halfcheetah-Med-Replay",
        'random': "Halfcheetah-Random",
    }


    fig,axes = plt.subplots(nrows = 1, ncols = 9, figsize = (25, 3),  sharex=True, sharey=True)
    axes[0].set_ylabel('Returns', fontsize = fs)

    for idx, ax in enumerate(axes):
        dataset = datasets[idx//3]

        ratio = ratios[idx%3]


        

        with open(f"dataset_infos/cluster_infos_{env}-{dataset}-{ratio:.2f}-v2.pkl", "rb") as f:
            infos = pkl.load(f)
        trajs = infos['mixed_trajectories']
        returns = [   np.sum(traj['rewards'])  for traj in trajs   ]
        # idx_sorted, returns_sorted = sort_by_values(  np.arange(len(returns)), np.array(returns)  )

        # kde = gaussian_kde(returns)
        # x_values = np.linspace(min(returns), max(returns), 1000)
        # density_values = kde(x_values)
        # ax.plot(x_values, density_values, color='blue')

        # if idx %3 == 2 :
        #     ax.axvline(x= max(returns) + 500, color='gray', linestyle='--')

        sns.kdeplot(returns, ax = ax, fill=True, linewidth = 1.0, color="#e5541d")
        if idx % 3 == 1:
            ax.set_title(f'{name[dataset]}-0.1, 0.2, 0.3', fontsize = fs)    
        # if idx % 3 == 2:
        #     ax.text(0.5, 0.01,f"Halfcheetah-{name[dataset]}", ha='center', va='bottom')
    plt.tight_layout(pad = 0.05)
    plt.subplots_adjust(bottom=0.1, left = 0.04)
    # x_min, x_max = plt.gca().get_xlim() 
    # positions = [ax.get_position() for ax in axes]
    # for idx in [2,5,7]:
    #     pos = positions[idx]
    #     x = (pos.x0 + pos.x1) / 2
    #     y = pos.y0 - 0.25  # 在子图下方适当位置添加文字
    #     fig.text(x, y, f"Halfcheetah-{name[dataset]}", ha='center', va='top')
    # plt.savefig("my.png", dpi = 600)
    plt.savefig("pdf_values_miaxed_halfcheetah.pdf", format = 'pdf')
    return



def visual_values_pdf_sep(env, dataset):

    ratios = [0.1,0.2,0.3]
    datasets = ["medium", "medium-replay", "random"]
    name = {
        'medium': "(a) Halfcheetah-Medium",
        'medium-replay': "(b) Halfcheetah-Med-Replay",
        'random': "(c) Halfcheetah-Random",
    }


    name = {
        'medium': "(a) Halfcheetah-Medium",
        'medium-replay': "(b) Halfcheetah-Med-Replay",
        'random': "(c) Halfcheetah-Random",
    }


    fig,axes = plt.subplots(nrows = 1, ncols = 3, figsize = (8, 3),  sharex=True, sharey=True)

    if dataset == "medium":
        axes[0].set_ylabel('Returns', fontsize = fs)

    for idx, ax in enumerate(axes):

        ratio = ratios[idx%3]

        with open(f"dataset_infos/cluster_infos_{env}-{dataset}-{ratio:.2f}-v2.pkl", "rb") as f:
            infos = pkl.load(f)
        trajs = infos['mixed_trajectories']
        returns = [   np.sum(traj['rewards'])  for traj in trajs   ]
        # idx_sorted, returns_sorted = sort_by_values(  np.arange(len(returns)), np.array(returns)  )

        # kde = gaussian_kde(returns)
        # x_values = np.linspace(min(returns), max(returns), 1000)
        # density_values = kde(x_values)
        # ax.plot(x_values, density_values, color='blue')

        # if idx %3 == 2 :
        #     ax.axvline(x= max(returns) + 500, color='gray', linestyle='--')

        sns.kdeplot(returns, ax = ax, fill=True, linewidth = 1.0, color="#e5541d")
        # if idx % 3 == 1:
        #     ax.set_title(f'{name[dataset]}-0.1, 0.2, 0.3', fontsize = fs, y =-0.4)    
        # if idx % 3 == 2:
        #     ax.text(0.5, 0.01,f"Halfcheetah-{name[dataset]}", ha='center', va='bottom')
        ax.tick_params(axis='both', which='major', labelsize=fs  - 7)
    plt.tight_layout(pad = 0.05)
    plt.subplots_adjust(bottom=0.2, left = 0.1 if dataset=="medium" else 0.06,  right=0.98)
    # plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    # x_min, x_max = plt.gca().get_xlim() 
    # positions = [ax.get_position() for ax in axes]
    # for idx in [2,5,7]:
    #     pos = positions[idx]
    #     x = (pos.x0 + pos.x1) / 2
    #     y = pos.y0 - 0.25  # 在子图下方适当位置添加文字
    #     fig.text(x, y, f"Halfcheetah-{name[dataset]}", ha='center', va='top')
    # plt.savefig("my.png", dpi = 600)
    plt.savefig(f"pdf_values_miaxed_halfcheetah_{dataset}.png", dpi = 1000,  format = 'png')
    return

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def visual_values_pdf_v2(env):

    ratios = [0.1,0.2,0.3]
    datasets = ["medium", "medium-replay", "random"]
    name = {
        'medium': ["Medium", "Med-Exp (Ratio=0.1)", "Med-Exp (Ratio=0.2)", "Med-Exp (Ratio=0.3)"],
        'medium-replay': ["Medium-Replay", "MR-Exp (Ratio=0.1)", "MR-Exp (Ratio=0.2)", "MR-Exp (Ratio=0.3)"],
        'random': ["Random", "Rand-Exp (Ratio=0.1)", "Rand-Exp (Ratio=0.2)", "Rand-Exp (Ratio=0.3)"],
    }


    # fig,axes = plt.subplots(nrows = 1, ncols = 9, figsize = (25, 3),  sharex=True, sharey=True)
    # fig = plt.figure(constrained_layout = True, figsize=(25, 3))
    fig = plt.figure( figsize= (22,2.9)  )
    outer = gridspec.GridSpec( 1,3, wspace=0.1  )

    axes_outer = fig.subfigures(1,3)

    # axes[0].set_ylabel('Returns', fontsize = fs)
    for idx_dataset, ax_outer in enumerate(axes_outer):
        inner = gridspec.GridSpecFromSubplotSpec( 1,3, subplot_spec=outer[idx_dataset], wspace=0.001 )

        dataset = datasets[idx_dataset]

        axes_inner = ax_outer.subplots(nrows=1, ncols=3, sharey = True)
        for idx_ratio, ax in enumerate(axes_inner):
            ratio = ratios[idx_ratio]
            this_ax = plt.Subplot( fig, inner[idx_ratio] )

        

            with open(f"dataset_infos/cluster_infos_{env}-{dataset}-{ratio:.2f}-v2.pkl", "rb") as f:
                infos = pkl.load(f)
            trajs = infos['mixed_trajectories']
            returns = [   np.sum(traj['rewards'])  for traj in trajs   ]
            idx_sorted, returns_sorted = sort_by_values(  np.arange(len(returns)), np.array(returns)  )

            returns_to_vis = list(returns_sorted[::len(returns)//100])
            padding_length = 100 - len(returns_to_vis)
            if padding_length > 0:
                for i in range(padding_length):
                    returns_to_vis.append(returns_to_vis[-1])
            elif padding_length < 0:
                returns_to_vis = returns_to_vis[:padding_length]
            else:
                print("No need to padding")

            ax.bar(range(len(returns_to_vis)), returns_to_vis)
            ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
            # kde = gaussian_kde(returns)
            # x_values = np.linspace(min(returns), max(returns), 1000)
            # density_values = kde(x_values)
            # ax.plot(x_values, density_values, color='blue')

            # if idx %3 == 2 :
            #     ax.axvline(x= max(returns) + 500, color='gray', linestyle='--')

            # sns.kdeplot(returns, ax = ax, fill=True, linewidth = 1.0, color="#e5541d")

            # if idx_ratio == 1:
            ax.set_title(f'{name[dataset][idx_ratio]}', fontsize = fs, y = 0.0)    
            fig.add_subplot(this_ax)
        # if idx % 3 == 2:
        #     ax.text(0.5, 0.01,f"Halfcheetah-{name[dataset]}", ha='center', va='bottom')
    # plt.tight_layout(pad = 0.05)
            
    
    plt.subplots_adjust(bottom=0.25, left = 0.1)
    
    # x_min, x_max = plt.gca().get_xlim() 
    # positions = [ax.get_position() for ax in axes]
    # for idx in [2,5,7]:
    #     pos = positions[idx]
    #     x = (pos.x0 + pos.x1) / 2
    #     y = pos.y0 - 0.25  # 在子图下方适当位置添加文字
    #     fig.text(x, y, f"Halfcheetah-{name[dataset]}", ha='center', va='top')
    # plt.savefig("my.png", dpi = 600)
    plt.savefig("pdf_values_miaxed_halfcheetah.pdf", format = 'pdf')
    return


def visual_values_pdf_v2(env):
    label_pos = -0.15
    ratios = [0.0, 0.1,0.2,0.3]
    datasets = ["medium", "medium-replay", "random"]
    name = {
        'medium': ["Medium", "Med-Exp (Ratio=0.1)", "Med-Exp (Ratio=0.2)", "Med-Exp (Ratio=0.3)"],
        'medium-replay': ["Medium-Replay", "MR-Exp (Ratio=0.1)", "MR-Exp (Ratio=0.2)", "MR-Exp (Ratio=0.3)"],
        'random': ["Random", "Rand-Exp (Ratio=0.1)", "Rand-Exp (Ratio=0.2)", "Rand-Exp (Ratio=0.3)"],
    }


    fig,axes = plt.subplots(nrows = 3, ncols = 4, figsize = (28, 23),  sharex=True, sharey=True)

    # axes[0].set_ylabel('Returns', fontsize = fs)
    for idx_dataset, ax_outer in enumerate(datasets):
        dataset = datasets[idx_dataset]
        if idx_dataset == 2:
            label_pos *= 1.7
        for idx_ratio, ratio in enumerate(ratios):
            this_ax = axes[idx_dataset][idx_ratio]

            if ratio == 0:
                filepath = f"dataset_infos/cluster_infos_{env}-{dataset}-v2.pkl"
            else:
                filepath = f"dataset_infos/cluster_infos_{env}-{dataset}-{ratio:.2f}-v2.pkl"
            with open( filepath , "rb") as f:
                infos = pkl.load(f)
            trajs = infos['mixed_trajectories']
            returns = [   np.sum(traj['rewards'])  for traj in trajs   ]
            idx_sorted, returns_sorted = sort_by_values(  np.arange(len(returns)), np.array(returns)  )

            returns_to_vis = list(returns_sorted[::len(returns)//100])
            padding_length = 100 - len(returns_to_vis)
            if padding_length > 0:
                for i in range(padding_length):
                    returns_to_vis.append(returns_to_vis[-1])
            elif padding_length < 0:
                returns_to_vis = returns_to_vis[:padding_length]
            else:
                print("No need to padding")

            this_ax.bar(range(len(returns_to_vis)), returns_to_vis)
            # this_ax.ticklabel_format(style='sci', axis='both', scilimits=(0, 0),)
            

            
            this_ax.set_title(f'{name[dataset][idx_ratio]}', fontsize = fs, y = label_pos)    
            this_ax.tick_params(axis='x', labelsize=fs)  # 横坐标
            this_ax.tick_params(axis='y', labelsize=fs)  # 纵坐标

            fig.add_subplot(this_ax)
        # if idx % 3 == 2:
        #     ax.text(0.5, 0.01,f"Halfcheetah-{name[dataset]}", ha='center', va='bottom')
    plt.tight_layout(pad = 0.05)    
    # x_min, x_max = plt.gca().get_xlim() 
    # positions = [ax.get_position() for ax in axes]
    # for idx in [2,5,7]:
    #     pos = positions[idx]
    #     x = (pos.x0 + pos.x1) / 2
    #     y = pos.y0 - 0.25  # 在子图下方适当位置添加文字
    #     fig.text(x, y, f"Halfcheetah-{name[dataset]}", ha='center', va='top')
    # plt.savefig("my.png", dpi = 600)
    plt.subplots_adjust(hspace=0.2)
    plt.subplots_adjust(wspace=0.1)
    plt.subplots_adjust(left=0.07, right=0.98, top  =0.98, bottom=0.1)
    plt.savefig("pdf_values_miaxed_appendix.pdf", format = 'pdf')
    # plt.savefig("pdf_values_miaxed_appendix.png", format = 'png', dpi = 200)
    return

if __name__ == "__main__":

    # for dataset in ["medium", "medium-replay", "random"]:
    visual_values_pdf_v2("halfcheetah")