import tensorboard
from torch.utils.tensorboard import SummaryWriter
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
from diffuser.datasets.d4rl import *
os.environ['OPENBLAS_NUM_THREADS'] = '1'

path = "/home/$USER/sync/CDiffuser/kitchen/logs"

pattern = re.compile(r"\[STEP (\d+)/\d+.*Diff_loss: +([\d.]+)")
pattern_step = re.compile(r"STEP (\d+)/\d+")
loss_patterns = {
    "a0_loss":re.compile(r"a0_loss: +([\d.-]+)"),
    "corr": re.compile(r"corr: +([\d.-]+)"),
    "Value_loss": re.compile(r"Value_loss: +([\d.-]+)"),
    "Diff_loss": re.compile(r"Diff_loss: +([\d.-]+)"),
    "Stable_loss": re.compile(r"Stable_loss: +([\d.-]+)"),
    "Loss": re.compile(r"Loss: +([\d.-]+)"),

}

from datetime import datetime
class MyLogger():
    def __init__(self, logfile = "log.log") -> None:
        self.logfile = logfile
        pass
    def log(self, info, flush = True):
        current_time = datetime.now()
        info_with_time = f"[{current_time}]  {info}"
        cmd = f"echo \"{info_with_time}\" >> {self.logfile}"
        os.system(cmd)
        print(info_with_time, flush=flush)
        return

class Analyzer:
    def __init__(self) -> None:
        pass
    def fit(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, copy.deepcopy(v))


    def showState(self):
        for rec in ["traj","s"]:
            for indicator in ['ad']:
                self.vis_values(rec, indicator)
        return
    
    def sort_by_values(self, to_sort, values):
        inds = np.argsort(values)[::-1]
        if to_sort:
            to_sort = to_sort[inds]
        values = values[inds]
        return to_sort, values

    def vis_values(self,  plan = "sa", indicator = "a" ):

        savedir = "figs"
        recores = []

        recores = np.array(recores)
        umap_reducer = umap.UMAP(n_components=2)
        embedding = umap_reducer.fit_transform(recores)


        values = np.array(123).reshape(-1,1)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        print("[build_values_actionstate] Adding scatter.")
        ax.scatter(embedding[:, 0], embedding[:, 1], values[:, 0], c=values[:, 0], cmap='viridis', s = 2)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Values')

        base_filename = os.path.join( savedir, base_filename )
        plt.savefig(f"{base_filename}.png", dpi=500)

        ax.view_init(elev=90, azim=0)
        plt.savefig(f"{base_filename}_top.png", dpi=500)

        ax.view_init(elev=270, azim=180)
        plt.savefig(f"{base_filename}_bottom.png", dpi=500)

        # 排序后可视化成柱状图
        rewards_record_sort = copy.deepcopy(recores)
        _, rewards_record_sort = self.sort_by_values(None,  np.array(rewards_record_sort).reshape(-1))
        
        rewards_record_sort_sample = rewards_record_sort[::rewards_record_sort.shape[0]//10]
        plt.clf()
        plt.clf()

        plt.bar(range(rewards_record_sort_sample.shape[0]), rewards_record_sort_sample)
        plt.savefig(f"{base_filename}_values.pdf", format = "pdf")
        plt.cla()

        return
    




def convert_log(log_dir):
    Configs.logger.log(f"Converting logfile to tensorboard.({log_dir})")
    shutil.rmtree(os.path.join(log_dir, "boards", "train"), ignore_errors=True)
    writer = SummaryWriter(log_dir =  os.path.join(log_dir, "boards", "train"))
    logfile =  os.path.join(log_dir, "training_log.log")
    if not os.path.exists(logfile): Configs.logger.log(f"No such file:{logfile}")
    with open(logfile, "r") as f:
        try:
            for line in f:
                
                if not re.compile("TensorboardInfos").search(line):
                    continue
                infos = line.split("|")[1:]
                step = int(infos[0])
                for metrics in  infos[1:]:
                    splited = metrics.split("@")
                    writer.add_scalar(splited[0], float(splited[1]), step)
            Configs.logger.log(f"Convert Done.")
        except Exception as e:
                Configs.logger.log(f"Error occured during converting logfile:{logfile}")
                Configs.logger.log(e)

def build_renderer(env_name):
    Configs.device = "cuda:0"
    render_config = utils.Config(
        "utils.MuJoCoRenderer",
        savepath=None,
        env=env_name
    )
    return render_config()

def render_a_state(renderer, state, savepath):

    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    state = state.reshape(1,1,-1)
    plt.clf()
    plt.cla()
    images = renderer.composite(savepath = None, paths = state, dim = (512,512))
    plt.imshow(images)
    plt.savefig(os.path.join(savepath, "state.png"), dpi = 200  )

    return

def convert_eval(log_dir):
    Configs.logger.log(f"Converting logfile to tensorboard.({log_dir})")
    # writer = SummaryWriter(log_dir = log_dir)

    logfiles =  glob.glob(os.path.join(log_dir, "eval_log_LOAD*.log"))

    for logfile in logfiles:
        log_dir = logfile.split(".log")[0]
        
        log_dir = log_dir.replace(log_dir.split("/")[-1],  os.path.join("boards", "eval", log_dir.split("/")[-1]) )
        
        # items = []
        # writer = SummaryWriter(log_dir = )
        # base_name = 

        # if not os.path.exists(logfile): Configs.logger.log(f"No such file:{logfile}")
        with open(logfile, "r") as f:
            try:
                for line in f:
                    if not re.compile("####TensorboardInfos####").search(line):
                        continue
                    infos = line.split("|")[1:]
                    item = infos[0]
                    evalseed = infos[1].split("@")[-1]
                    step = infos[2].split("@")[-1]
                    prefix = f"EvalSeed{evalseed}/ModelStep{step}/"
                    writer = SummaryWriter(os.path.join(log_dir, prefix))
                    logs = infos[3].split("#")
                    for rec in  logs:
                        splited = rec.split("@")
                        writer.add_scalar(item, float(splited[1]),  float(splited[0]))
                    writer.close()
            except Exception as e:
                Configs.logger.log(f"Error occured during converting logfile:{logfile}")
                Configs.logger.log(e)
                pass
        
    Configs.logger.log(f"Convert Done.")

def convert_evalresults(log_dir):
    Configs.logger.log(f"Converting logfile to tensorboard.({log_dir})")
    writer = SummaryWriter(log_dir = log_dir)
    # board_ = os.path.join(log_dir)
    logfiles =  glob.glob(os.path.join(log_dir, "evalresults_LOAD*.log"))
    all= []
    for logfile in logfiles:
        log_dir = logfile.split(".log")[0]
        
        log_dir = log_dir.replace(log_dir.split("/")[-1],  os.path.join("boards", "eval", log_dir.split("/")[-1]) )
        
        # items = []
        # writer = SummaryWriter(log_dir = )
        # base_name = 

        # if not os.path.exists(logfile): Configs.logger.log(f"No such file:{logfile}")
        with open(logfile, "r") as f:
            try:
                for line in f:
                    if not re.compile("####TensorboardInfos####").search(line):
                        continue
                    infos = line.split("|")[1:]
                    score = float(infos[0])
                    all.append(score)
            except Exception as e:
                Configs.logger.log(f"Error occured during converting logfile:{logfile}")
                Configs.logger.log(e)
                pass
    all = np.array(all)
    writer.add_text( "results", f"Average:{np.mean(all)}, Std:{np.std(all)}, evaled {len(all)} times.")
    Configs.logger.log(f"Convert Done.")

def convert_sepeval(log_dir, final = False):
    convert_sepeval_temp(log_dir, final)
    return
    Configs.logger.log(f"Converting logfile to tensorboard.({log_dir})")
    
    if not final:
        logfile =  os.path.join(log_dir, "eval_log.log")
        board_dir =  os.path.join(log_dir, "boards", "eval")
        os.system(f"rm -rf {board_dir}/events*")
        writer = SummaryWriter(log_dir = board_dir)
    else:
        board_dir =  os.path.join(log_dir, "boards", "eval_final")
        os.system(f"rm -rf {board_dir}/events*")
        writer = SummaryWriter(log_dir = board_dir)
        logfile =  os.path.join(log_dir, "eval_log_final.log")
    if not os.path.exists(logfile): Configs.logger.log(f"No such file:{logfile}")
    with open(logfile, "r") as f:
        try:
            for line in f:
                
                if not re.compile("###SepEval###").search(line):
                    continue
                infos = line.split("|")[1:]
                step = int(infos[0])
                for metrics in  infos[1:]:
                    splited = metrics.split(":")
                    writer.add_scalar(splited[0], float(splited[1]), step)
            
        except Exception as e:
                Configs.logger.log(f"Error occured during converting logfile:{logfile}")
                Configs.logger.log(e)
    Configs.logger.log(f"Convert Done.")




def convert_sepeval_temp(log_dir, final = False):

    Configs.logger.log(f"Converting logfile to tensorboard.({log_dir})")
    
    if not final:
        logfile =  os.path.join(log_dir, "eval_log.log")
        board_dir =  os.path.join(log_dir, "boards", "eval")
        os.system(f"rm -rf {board_dir}/events*")
        writer = SummaryWriter(log_dir = board_dir)
    else:
        board_dir =  os.path.join(log_dir, "boards", "eval_final")
        os.system(f"rm -rf {board_dir}/events*")
        writer = SummaryWriter(log_dir = board_dir)
        logfile =  os.path.join(log_dir, "eval_log_final.log")
    if not os.path.exists(logfile): Configs.logger.log(f"No such file:{logfile}")
    all = []
    with open(logfile, "r") as f:
        try:
            for line in f:
                
                if not re.compile("###SepEval###").search(line):
                    continue
                infos = line.split("|")[1:]
                # step = int(infos[0])
                # for metrics in  infos[1:]:
                #     splited = metrics.split(":")
                #     writer.add_scalar(splited[0], float(splited[1]), step)
                score = float(infos[1].split(':')[1])
                all.append(score)
        except Exception as e:
                Configs.logger.log(f"Error occured during converting logfile:{logfile}")
                Configs.logger.log(e)
    all = np.array(all)
    writer.add_text( "results", f"Average:{np.mean(all)}, Std:{np.std(all)}")
    Configs.logger.log(f"Convert Done.")

import random
import string
import time
def get_token():
    current_timestamp = time.time()
    state = random.getstate()
    random.seed(current_timestamp)
    letters = string.ascii_letters + string.digits
    token = ''.join(random.choice(letters) for _ in range(16))
    random.setstate(state)
    return token

from diffuser.datasets.d4rl import sequence_dataset, sequence_dataset_plain, load_environment,sequence_dataset_mix

def preprocess_dataset(dataset):
    processed_dataset = { }
    for k in dataset.keys():
        if 'metadata' in k:
            processed_dataset[k] = dataset[k]
        else:
            processed_dataset[k] = []
    
    rewards = []
    for cell in dataset:
        for k in cell.keys():
            rewards.append(np.sum(cell['rewards']))
    
    all_data = []

def sort_by_values( to_sort, values):
    inds = np.argsort(values)[::-1]
    to_sort = to_sort[inds]
    values = values[inds]
    return to_sort, values

def generate_expert_info( vis = False):

    for env_name in ["hopper","halfcheetah", "walker2d"]:
        print(f"{env_name}")
        expert_env = load_environment(f"{env_name}-expert-v2")
        expert_trajs, expert_data = sequence_dataset_mix(expert_env)


        max_path_length = expert_env.max_episode_steps
        vs_discount = 1e-4 ** (1/max_path_length)
        vs_discounts = vs_discount ** np.arange(max_path_length)[:, None]
        with open(f"value_of_states/{env_name}-expert-v2.pkl", "rb") as f:
            vlaues_of_states = pkl.load(f)

        # vlaues_of_states = calculate_vs(expert_data, max_path_length, vs_discounts)
        accumulated_rewards = [  traj['accumulated_reward'] for traj in expert_trajs   ]

        inds = np.argsort(accumulated_rewards)[::-1]

        if vis:
            if not os.path.exists("vis"):
                os.mkdir("vis")

            rewards_record_sort = np.array(accumulated_rewards)[inds]
            rewards_record_sort_sample = rewards_record_sort[::rewards_record_sort.shape[0]//100]
            plt.clf()

            plt.bar(range(rewards_record_sort_sample.shape[0]), rewards_record_sort_sample)
            plt.savefig(f"vis/{env_name}-expert-v2.png", format = "png", dpi = 200)
            plt.cla()


        
        expert_info = {
            'dataset': expert_data, 
            'trajs': expert_trajs, 
            'accumulated_rewards': accumulated_rewards, 
            'high_to_low': inds, 
            'vlaues_of_states': vlaues_of_states  
        }
        
        if not os.path.exists("expert_infos"):
            os.mkdir("expert_infos")
        with open(  f"expert_infos/expert_info_{env_name}.pkl"  , "wb"    ) as f:
            pkl.dump(expert_info, f)
    return


def generate_dataset_infos(env_name, dataset_name):

    print(f"{env_name}")
    name = f"{env_name}-{dataset_name}-v2"
    expert_env = load_environment(name)
    expert_trajs, expert_data = sequence_dataset_mix(expert_env)


    max_path_length = expert_env.max_episode_steps
    with open(f"value_of_states/{name}.pkl", "rb") as f:
        vlaues_of_states = pkl.load(f)

    accumulated_rewards = [  traj['accumulated_reward'] for traj in expert_trajs   ]

    inds = np.argsort(accumulated_rewards)[::-1]

    if not os.path.exists("vis"):
        os.mkdir("vis")

    rewards_record_sort = np.array(accumulated_rewards)[inds]
    rewards_record_sort_sample = rewards_record_sort[::rewards_record_sort.shape[0]//100]
    plt.clf()

    plt.bar(range(rewards_record_sort_sample.shape[0]), rewards_record_sort_sample)
    plt.savefig(f"vis/{name}.png", format = "png", dpi = 200)
    plt.cla()


    
    expert_info = {
        'dataset': expert_data,
        'trajs': expert_trajs, 
        'accumulated_rewards': accumulated_rewards,
        'high_to_low': inds, 
        'vlaues_of_states': vlaues_of_states  
    }
    
    if not os.path.exists("dataset_infos"):
        os.mkdir("dataset_infos")
    with open(  f"dataset_infos/dataset_info_{name}.pkl"  , "wb"    ) as f:
        pkl.dump(expert_info, f)

    

    return


def determain_env(env_full_name):

    what_env_is_it = {
        'maze2d': 'maze2d' in env_full_name,
        'antmaze': 'antmaze-' in env_full_name,
        'adroit': 'pen-' in env_full_name or 'hammer-' in env_full_name or 'door-' in env_full_name or 'relocate-' in env_full_name,
        'gym': '-v2' in env_full_name,
        'kitchen': 'kitchen' in env_full_name
    }

    return what_env_is_it

from tqdm import tqdm


def calculate_vs(dataset,max_path_length,vs_discounts):
    
    N = dataset['rewards'].shape[0]
    terminal_points = list(np.where(dataset['terminals'])[0])
    # terminal_points.append(32)
    if not dataset['terminals'][-1]:
        padding = np.zeros(max_path_length)
        dataset['rewards'] = np.concatenate((dataset['rewards'], padding), -1)
    terminal_points.append(N + max_path_length)        
    vlaues_of_states = []
    for i in tqdm(range(N)):
        if i > terminal_points[0]:
            terminal_points = terminal_points[1:]
        end = min( i + max_path_length, terminal_points[0]   )
        rewards_si = dataset['rewards'][i:end]
        padding_length = max_path_length - len(rewards_si)
        if padding_length > 0:
            padding = np.ones(padding_length)*-1
            rewards_si = np.concatenate((rewards_si, padding), -1)

        value_si = np.sum(   vs_discounts * rewards_si   )
        vlaues_of_states.append(value_si)
    
    return vlaues_of_states


def calculate_vs_kitchen(name):
    
    env = load_environment(name)
    max_path_length_env = env.max_episode_steps
    discount = pow(0.001, 1/max_path_length_env)
    discounts = discount ** np.arange(max_path_length_env)[:, None]

    trajs, dataset = sequence_dataset_mix_kitchen(env)
    max_path_length = np.max([  len(traj['observations']) for traj in trajs ] )

    N = len(dataset['terminals'])
    # dataset = self.plain_datset
    terminal_points = list(np.where(dataset['terminals'])[0])
    # terminal_points.append(32)
    if not dataset['terminals'][-1]:
        padding = np.ones(max_path_length)*dataset['rewards'][-1] 
        dataset['rewards'] = np.concatenate((dataset['rewards'], padding), -1)
    terminal_points.append(N + max_path_length)        
    vlaues_of_states = []
    for i in tqdm(range(N)):
        if i > terminal_points[0]:
            terminal_points = terminal_points[1:]
        end = min( i + max_path_length, terminal_points[0]   )
        rewards_si = dataset['rewards'][i:end]
        padding_length = max_path_length - len(rewards_si)
        if padding_length > 0:
            padding = np.ones(padding_length)*dataset['rewards'][end]
            rewards_si = np.concatenate((rewards_si, padding), -1)

        value_si = np.sum(   discounts * rewards_si   )
        vlaues_of_states.append(value_si)
    
    return vlaues_of_states



def vis_rewards(env_class, env_name):
    print(f"Processing {env_name}.")
    save_folder = f"vis/instant_reward/{env_class}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # for env_name in ["halfcheetah", "hopper", "walker2d"]:
    #     for dataset in ["expert", "medium-expert", "medium-replay","medium", "random"]:
    # print(f"{env_name}-{dataset}-v2")
    env = load_environment(env_name)
    trajs, _ = sequence_dataset_mix(env)

    # accumulated_rewards = [  traj['accumulated_reward'] for traj in trajs   ]
    inds = np.argsort(_['rewards'])[::-1]

    rewards_record_sort = np.array(_['rewards'])[inds]
    rewards_record_sort_sample = rewards_record_sort[::rewards_record_sort.shape[0]//100]
    plt.clf()

    plt.bar(range(rewards_record_sort_sample.shape[0]), rewards_record_sort_sample)
    tile_path = os.path.join(save_folder, f"{env_name}.png")
    plt.savefig(tile_path, format = "png", dpi = 300)
    plt.cla()

    return


def vis_accumulated_rewards(env_class, env_name):
    print(f"Processing {env_name}.")
    save_folder = f"vis/accumulated_rewards/{env_class}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # for env_name in ["halfcheetah", "hopper", "walker2d"]:
    #     for dataset in ["expert", "medium-expert", "medium-replay","medium", "random"]:
    # print(f"{env_name}-{dataset}-v2")
    env = load_environment(env_name)
    trajs, _ = sequence_dataset_mix(env)

    # accumulated_rewards = [  traj['accumulated_reward'] for traj in trajs   ]
    inds = np.argsort(_['rewards'])[::-1]

    rewards_record_sort = np.array(_['rewards'])[inds]
    rewards_record_sort_sample = rewards_record_sort[::rewards_record_sort.shape[0]//100]
    plt.clf()

    plt.bar(range(rewards_record_sort_sample.shape[0]), rewards_record_sort_sample)
    tile_path = os.path.join(save_folder, f"{env_name}.png")
    plt.savefig(tile_path, format = "png", dpi = 300)
    plt.cla()

    return



def vis_raw_rewards(env_class, env_name):
    print(f"Processing {env_name}.")
    save_folder = f"vis/raw_reward/{env_class}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # for env_name in ["halfcheetah", "hopper", "walker2d"]:
    #     for dataset in ["expert", "medium-expert", "medium-replay","medium", "random"]:
    # print(f"{env_name}-{dataset}-v2")
    env = load_environment(env_name)
    trajs, _ = sequence_dataset_mix(env)

    accumulated_rewards = [  traj['accumulated_reward'] for traj in trajs   ]
    inds = np.argsort(accumulated_rewards)[::-1]

    rewards_record_sort = np.array(accumulated_rewards)[inds]
    rewards_record_sort_sample = rewards_record_sort[::rewards_record_sort.shape[0]//100]
    plt.clf()

    plt.bar(range(rewards_record_sort_sample.shape[0]), rewards_record_sort_sample)
    tile_path = os.path.join(save_folder, f"{env_name}.png")
    plt.savefig(tile_path, format = "png", dpi = 50)
    plt.cla()

    return


def vis_mixed_rewards(ratio):
    for env_name in ["halfcheetah", "hopper", "walker2d"]:
        for dataset in ["medium-replay","medium", "random"]:
            with open(  f"mixed_datasets/{env_name}-{dataset}-mix-expert-{ratio:.2f}-v2.pkl"  , "rb"    ) as f:
                mixed_dataset = pkl.load(f)

            trajs = mixed_dataset['trajs']

            accumulated_rewards = [  traj['accumulated_reward'] for traj in trajs   ]
            inds = np.argsort(accumulated_rewards)[::-1]

            if not os.path.exists("vis"):
                os.mkdir("vis")

            rewards_record_sort = np.array(accumulated_rewards)[inds]
            rewards_record_sort_sample = rewards_record_sort[::rewards_record_sort.shape[0]//100]
            plt.clf()

            plt.bar(range(rewards_record_sort_sample.shape[0]), rewards_record_sort_sample)
            plt.savefig( f"vis/{env_name}-{dataset}-mix-expert-{ratio:.2f}-v2.png", format = "png", dpi = 200)
            plt.cla()

    return


def assert_dataset_infos():

    for env_name in ["halfcheetah", "hopper", "walker2d"]:
        expert_infos_file = f"expert_infos/expert_info_{env_name}.pkl" 
        with open(expert_infos_file, "rb") as f:
            expert_infos = pkl.load(f)
        len_of_expert = len(expert_infos['dataset']['observations'])
        for dataset_type in ["medium", "medium-replay", "random"]:
            name = f"{env_name}-{dataset_type}-v2"
            env = load_environment(name)
            trajs, dataset = sequence_dataset_mix(env)
            length_of_current = len(dataset['observations'])
            if len_of_expert >= length_of_current: 
                print(  f"{name} check passed." )
            else:
                print(  f"Expert: {len_of_expert} != {name}: {length_of_current}"  )
            


            continue
            
    return


def generate_mix_datasets(ratio = 0.01):

    for env_name in ["halfcheetah", "hopper", "walker2d"]:
        for dataset_type in ["medium", "medium-replay", "random"]:
            name = f"{env_name}-{dataset_type}-v2"
            env = load_environment(name)
            trajs, dataset = sequence_dataset_mix(env)


            # max_path_length = env.max_episode_steps
            # vs_discount = 1e-4 ** (1/max_path_length)
            # vs_discounts = vs_discount ** np.arange(max_path_length)[:, None]

            with open(f"value_of_states/{name}.pkl", "rb") as f:
                vlaues_of_states = pkl.load(f)
            # vlaues_of_states = calculate_vs(dataset, max_path_length, vs_discounts)

            
            expert_infos_file = f"expert_infos/expert_info_{env_name}.pkl" 
            with open(expert_infos_file, "rb") as f:
                expert_infos = pkl.load(f)
            expert_trajs = expert_infos['trajs']
            inds = expert_infos['high_to_low'] 

            nums_origional = len(trajs)
            nums_to_mix = int(nums_origional * ratio)
            for idx in inds[:nums_to_mix]:
                trajs.append(expert_trajs[idx])  
            
            mixed_dataset = {
                "trajs": trajs,
                "mixed_expert_inds": inds[:nums_to_mix],
                "expert_dataset": expert_infos,
                "plain_vlaues_of_states": vlaues_of_states
            }
            if not os.path.exists("mixed_datasets"):
                os.mkdir("mixed_datasets")
            with open(  f"mixed_datasets/{env_name}-{dataset_type}-mix-expert-{ratio:.2f}-v2.pkl"  , "wb"    ) as f:
                pkl.dump(mixed_dataset, f)

            
    return

def render_dataset(type_of_enc = "mujoco", name = "walker2d-medium-v2", render_path =  "renders"):

    name =  "kitchen-mixed-v0"

    env = load_environment(name)
    trajs, dataset = sequence_dataset_mix(env)

    renderer = MuJoCoRenderer(env)
    if "kitchen" in name:
        renderer = engine.MovableCamera(env.sim, 1920, 2560)

    terminal_idxs = np.where(dataset['terminals'])[0] if 'terminals' in dataset.keys() else []
    timeouts = np.where(dataset['timeouts'])[0] if 'timeouts' in dataset.keys() else []

    # tmouts = [  list(range(_-20,_+20)) for _ in  timeouts[:20]   ]
    render_groups = {
        "continue": [list(range(20))],
        "terminals": [ list(range(_-50,_+50)) for _ in  terminal_idxs[:5]   ] ,
        "timeouts": [ list(range(_-50,_+50)) for _ in  timeouts[:10] ] ,
    }

    trash = {
        "continue":  [0],
        "terminals": terminal_idxs[:5],
        "timeouts": timeouts[:10] ,
    }

    
    # render_groups = ["continue", "terminals", "tmouts"]

    for render_type in render_groups.keys():
        for index, sequence in enumerate(render_groups[render_type]):
            target_folder = os.path.join(render_path,render_type,  name , str(trash[render_type][index]))
            if not os.path.exists(target_folder): os.makedirs(target_folder)
            for state_id in sequence:
                render_state_idx(   name, dataset, state_id, renderer,  savepath = target_folder, ismaze = bool("maze2d" in name), ismujoco = bool("ant-" in name) , isantmaze = bool("antmaze" in name), env=env   )
                
    sys.exit(0)
def render_image(name, dataset, idx, renderer,  ismaze = False, ismujoco = False , isantmaze = False):

    render_kwargs = {
            'trackbodyid': 2,
            'distance': 1,
            'lookat': [0, -5, 2],
            'elevation': 0
        }
    
    renderer.env.set_state(dataset['infos/qpos'][idx], dataset['infos/qvel'][idx])
    if ismaze:
        renderer.env.set_target(dataset['infos/goal'][idx])
        renderer.env.set_marker()
    if ismujoco:
        for key, val in render_kwargs.items():
            if key == 'lookat':
                renderer.viewer.cam.lookat[:] = val[:]
            else:
                setattr(renderer.viewer.cam, key, val)

    if isantmaze:
        renderer.env.set_target(dataset['infos/goal'][idx])
    renderer.viewer.render(512,512)
    data = renderer.viewer.read_pixels(512,512, depth=False)
    image = data[::-1, :, :]
    return image

def render_image_kitchen(name, dataset, idx, renderer, env):
    assert os.environ['MUJOCO_GL'] == "egl"
    obs = dataset['observations'][idx]
    qpos = obs[:30]
    env.robot.reset(env, qpos, np.zeros_like(qpos))
    camera = engine.MovableCamera(env.sim, 1920, 2560)
    camera.set_pose(distance=2.2, lookat=[-0.2, .5, 2.], azimuth=70, elevation=-35)
    image = camera.render()
    return image


def render_state_idx(name, dataset, idx, renderer,  savepath = "./", ismaze = False, ismujoco = False , isantmaze = False, env = None):
    
    
    if not os.path.exists(savepath): os.makedirs(savepath)
    
    if not "kitchen" in name:
        image = render_image(name, dataset, idx, renderer,  ismaze, ismujoco , isantmaze)
    else:
        image = render_image_kitchen(name, dataset, idx, renderer, env)
    plt.figure(); plt.clf(); plt.cla(); plt.imshow(image)
    plt.savefig(os.path.join(savepath,  f"{name}-{idx}.png"), dpi = 200  )


def visual_vs(savepath, name, vlaues_of_states):
    plt.cla()
    plt.clf()
    plt.figure()
    sorted_vals = sorted(vlaues_of_states)
    slim = sorted_vals[::len(sorted_vals)//100]
    plt.bar(range(len(slim)),slim)
    save_path = os.path.join(savepath, f"vs_{name}.png")
    plt.savefig(save_path , dpi = 500  )
    return



def gen_value_of_states(name = "walker2d-medium-v2"):
    savepath = "value_of_states"
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    whichenv = determain_env(name)
    if whichenv['gym']:
        env = load_environment(name)
        trajs, dataset = sequence_dataset_mix(env)
        max_path_length = env.max_episode_steps
        vs_discount = 1e-4 ** (1/max_path_length)
        vs_discounts = vs_discount ** np.arange(max_path_length)[:, None]
        vlaues_of_states = calculate_vs(dataset, max_path_length, vs_discounts)
    elif whichenv['maze2d']:
        raise NotImplementedError
    elif whichenv['kitchen']:
        env = load_environment(name)
        trajs, dataset = sequence_dataset_mix_kitchen(env)
        vlaues_of_states = calculate_vs_kitchen(name)
        visual_vs(savepath, name, vlaues_of_states)
    else:
        raise NotImplementedError

    
    with open(  f"{savepath}/{name}.pkl"  , "wb"    ) as f:
        pkl.dump(vlaues_of_states, f)


def visual_embeddings(env, dataset, ratio):

    print(f"Visualizing {env} {dataset} {ratio}")

    visual_save_path = "vis/embeddings"
    if not os.path.exists(visual_save_path):
        os.makedirs(visual_save_path)
    with open(f"dataset_infos/dataset_info_{env}-{dataset}-v2.pkl", "rb") as f:
        dataset_infos = pkl.load(f)

    
    states = dataset_infos['dataset']['observations']
    values_of_states = dataset_infos['vlaues_of_states']
    assert len(states) == len(values_of_states)
    nums_of_states = len(states) 
    values_orig = values_of_states[::nums_of_states//1000]
    values_orig = np.array(values_orig)
    states_orig = states[::nums_of_states//1000]

    nums_of_states = len(states_orig)

    with open(f"dataset_infos/dataset_info_{env}-expert-v2.pkl", "rb") as f:
        dataset_infos_expert = pkl.load(f)

    nums_to_mix = max(int(ratio * nums_of_states), 100)
    mixed_idxs = dataset_infos_expert['high_to_low'][:nums_to_mix]

    to_mix_states = dataset_infos_expert['dataset']['observations'][mixed_idxs]
    to_mix_values = np.array(dataset_infos_expert['vlaues_of_states'])[mixed_idxs]

    mixed_states = np.concatenate((to_mix_states, states_orig), axis=0)
    mixed_values = np.concatenate((to_mix_values, values_orig), axis=0)



    plt.clf()
    plt.cla()
    recores = np.array(mixed_states)
    umap_reducer = umap.UMAP(n_components=2)
    embedding_all = umap_reducer.fit_transform(recores)

    

    embedding = embedding_all[nums_to_mix:]
    values = np.array(mixed_values).reshape(-1,1)[nums_to_mix:]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(min(mixed_values), max(mixed_values))
    ax.set_xlim(min(embedding[:, 0]), max(embedding[:, 0])  )
    ax.set_ylim(min(embedding[:, 1]), max(embedding[:, 1])  )
    print("[build_values_actionstate] Adding scatter.")
    ax.scatter(embedding[:, 0], embedding[:, 1], values[:, 0], c=values[:, 0], cmap='viridis', s = 2)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Values')
    base_filename = f"{env}-{dataset}-orig"
    base_filename = os.path.join( visual_save_path, base_filename )
    plt.savefig(f"{base_filename}.png", dpi=500)
    ax.view_init(elev=90, azim=0)
    plt.savefig(f"{base_filename}_top.png", dpi=500)
    ax.view_init(elev=270, azim=180)
    plt.savefig(f"{base_filename}_bottom.png", dpi=500)


    embedding = embedding_all[:nums_to_mix]
    values = np.array(mixed_values).reshape(-1,1)[:nums_to_mix]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(min(mixed_values), max(mixed_values))
    ax.set_xlim(min(embedding[:, 0]), max(embedding[:, 0])  )
    ax.set_ylim(min(embedding[:, 1]), max(embedding[:, 1])  )
    print("[build_values_actionstate] Adding scatter.")
    ax.scatter(embedding[:, 0], embedding[:, 1], values[:, 0], c=values[:, 0], cmap='viridis', s = 2)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Values')
    base_filename = f"{env}-{dataset}-mixed"
    base_filename = os.path.join( visual_save_path, base_filename )
    plt.savefig(f"{base_filename}.png", dpi=500)
    ax.view_init(elev=90, azim=0)
    plt.savefig(f"{base_filename}_top.png", dpi=500)
    ax.view_init(elev=270, azim=180)
    plt.savefig(f"{base_filename}_bottom.png", dpi=500)



    embedding = embedding_all
    values = np.array(mixed_values).reshape(-1,1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(min(mixed_values), max(mixed_values))
    ax.set_xlim(min(embedding[:, 0]), max(embedding[:, 0])  )
    ax.set_ylim(min(embedding[:, 1]), max(embedding[:, 1])  )
    print("[build_values_actionstate] Adding scatter.")
    ax.scatter(embedding[:, 0], embedding[:, 1], values[:, 0], c=values[:, 0], cmap='viridis', s = 2)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Values')
    base_filename = f"{env}-{dataset}-all"
    base_filename = os.path.join( visual_save_path, base_filename )
    plt.savefig(f"{base_filename}.png", dpi=500)
    ax.view_init(elev=90, azim=0)
    plt.savefig(f"{base_filename}_top.png", dpi=500)
    ax.view_init(elev=270, azim=180)
    plt.savefig(f"{base_filename}_bottom.png", dpi=500)


    return


def visual_embeddings_2d(env, dataset, ratio):

    print(f"Visualizing {env} {dataset} {ratio}")

    visual_save_path = "vis/embeddings"
    if not os.path.exists(visual_save_path):
        os.makedirs(visual_save_path)
    with open(f"dataset_infos/dataset_info_{env}-{dataset}-v2.pkl", "rb") as f:
        dataset_infos = pkl.load(f)

    
    states = dataset_infos['dataset']['observations']
    values_of_states = dataset_infos['vlaues_of_states']
    assert len(states) == len(values_of_states)
    nums_of_states = len(states) 
    values_orig = values_of_states[::nums_of_states//1000]
    values_orig = np.array(values_orig)
    states_orig = states[::nums_of_states//1000]

    nums_of_states = len(states_orig)
    plt.clf()
    plt.cla()

    cmap = plt.cm.jet  

    
    



    with open(f"dataset_infos/dataset_info_{env}-expert-v2.pkl", "rb") as f:
        dataset_infos_expert = pkl.load(f)

    nums_to_mix = max(int(ratio * nums_of_states), 100)
    mixed_idxs = dataset_infos_expert['high_to_low'][:nums_to_mix]

    to_mix_states = dataset_infos_expert['dataset']['observations'][mixed_idxs]
    to_mix_values = np.array(dataset_infos_expert['vlaues_of_states'])[mixed_idxs]

    mixed_states = np.concatenate((to_mix_states, states_orig), axis=0)
    mixed_values = np.concatenate((to_mix_values, values_orig), axis=0)

    norm = Normalize(vmin=min(mixed_values), vmax=max(mixed_values))


    plt.clf()
    plt.cla()
    recores = np.array(mixed_states)
    umap_reducer = umap.UMAP(n_components=2)
    embedding_all = umap_reducer.fit_transform(recores)

    

    plt.figure()

    data = embedding_all[:nums_to_mix]
    labels = mixed_values[:nums_to_mix]
    plt.scatter(data[:, 0], data[:, 1], cmap='viridis', marker = 'x', s = 30, c=cmap(norm(labels)))

    data = embedding_all[nums_to_mix:]
    labels = mixed_values[nums_to_mix:]
    plt.scatter(data[:, 0], data[:, 1], cmap='viridis', marker = 's', s = 5, c=cmap(norm(labels)))

    base_filename = f"{env}-{dataset}-{ratio:.2f}"
    base_filename = os.path.join( visual_save_path, base_filename )

    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='Value')

    plt.savefig(f"{base_filename}_2d.png", dpi = 600)

    return



def visual_precomputed(env, dataset, ratio, metrics):

    print(f"Visualizing {env} {dataset} {ratio}")
    # metrics = "canberra"
    visual_save_path = f"vis/embeddings_{metrics}"
    if not os.path.exists(visual_save_path):
        os.makedirs(visual_save_path)
    with open(f"dataset_infos/dataset_info_{env}-{dataset}-v2.pkl", "rb") as f:
        dataset_infos = pkl.load(f)

    
    states = dataset_infos['dataset']['observations']
    values_of_states = dataset_infos['vlaues_of_states']
    assert len(states) == len(values_of_states)
    nums_of_states = len(states) 
    values_orig = values_of_states[::nums_of_states//1000]
    values_orig = np.array(values_orig)
    states_orig = states[::nums_of_states//1000]

    nums_of_states = len(states_orig)
    plt.clf()
    plt.cla()

    cmap = plt.cm.jet  

    
    



    with open(f"dataset_infos/dataset_info_{env}-expert-v2.pkl", "rb") as f:
        dataset_infos_expert = pkl.load(f)

    nums_to_mix = max(int(ratio * nums_of_states), 100)
    mixed_idxs = dataset_infos_expert['high_to_low'][:nums_to_mix]

    to_mix_states = dataset_infos_expert['dataset']['observations'][mixed_idxs]
    to_mix_values = np.array(dataset_infos_expert['vlaues_of_states'])[mixed_idxs]

    mixed_states = np.concatenate((to_mix_states, states_orig), axis=0)
    mixed_values = np.concatenate((to_mix_values, values_orig), axis=0)

    # 创建归一化对象
    norm = Normalize(vmin=min(mixed_values), vmax=max(mixed_values))


    plt.clf()
    plt.cla()
    recores = np.array(mixed_states)
    # hamming_dist = pdist(recores, metric='hamming')
    manhattan_distances = cdist(recores, recores, metric=metrics)
    # hamming_dist_matrix = squareform(manhattan_distances)

    adj = cosine_distances(recores)
    umap_reducer = umap.UMAP(n_components=2, metric='precomputed')
    embedding_all = umap_reducer.fit_transform(manhattan_distances)

    # umap_dir = f"vis/umap_obj" 
    # with open( f"vis/saved"  )

    plt.figure()

    data = embedding_all[:nums_to_mix]
    labels = mixed_values[:nums_to_mix]
    plt.scatter(data[:, 0], data[:, 1], cmap='viridis', marker = 'x', s = 30, c=cmap(norm(labels)))

    data = embedding_all[nums_to_mix:]
    labels = mixed_values[nums_to_mix:]
    plt.scatter(data[:, 0], data[:, 1], cmap='viridis', marker = 's', s = 5, c=cmap(norm(labels)))

    base_filename = f"{env}-{dataset}-{ratio:.2f}"
    base_filename = os.path.join( visual_save_path, base_filename )


    # with open(f"{base_filename}_umap.pkl", "wb") as f:
    #     pkl.dump(umap_reducer)

    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='Value')

    plt.savefig(f"{base_filename}_2d.png", dpi = 600)

    return

def generate_distances(env, dataset, ratio = 0.1, k = 3,  metrics = "canberra"):  
    
    distance_save_path = f"dataset_infos"
    if not os.path.exists(distance_save_path):
        os.makedirs(distance_save_path)
    with open(f"{distance_save_path}/dataset_info_{env}-{dataset}-v2.pkl", "rb") as f:
        dataset_infos = pkl.load(f)
    high_to_low = dataset_infos['high_to_low']
    do_not_select_me = len(dataset_infos['dataset']['observations']) - 1
    nums_to_mixin =  int(len(high_to_low)*ratio) + 5  
    mixin_idxs = high_to_low[:nums_to_mixin]
    
    tomix_state = []
    tomix_vs = []
    observation_idxs = []
    # next_states = []

    for i, idx in enumerate(mixin_idxs):
        tomix_state_in_traj = list(dataset_infos['trajs'][idx]['observations']) 
        trajectory_start_idx, trajectory_end_idx = dataset_infos['trajs'][idx]['start'],dataset_infos['trajs'][idx]['end']
        if trajectory_end_idx + 1 == trajectory_end_idx + len(tomix_state_in_traj):
            print("wtf2")
        # assert trajectory_end_idx + 1 == trajectory_end_idx + len(tomix_state_in_traj)
        tomix_vs_idx = dataset_infos['vlaues_of_states'][trajectory_start_idx: trajectory_end_idx + 1  ] 
        tomix_state += tomix_state_in_traj 
        tomix_vs += tomix_vs_idx           
        observation_idxs += list(range(trajectory_start_idx, trajectory_end_idx + 1)) 
        # next_states += list(dataset_infos['trajs'][idx]['next_observations'])

    tomix_state_reshape = []
    # next_states_reshape = []
    for idx, state in enumerate(tomix_state):
        tomix_state_reshape.append(state.reshape(1,-1))

    # for idx, state in enumerate(next_states):
    #     next_states_reshape.append(state.reshape(1,-1))
    
    states = np.concatenate(tomix_state_reshape,0)
    topk_distances = []
    # distances = []
    tok1k_idxs = []
    for i in tqdm(range(len(states)) ):
        x = states[i:i+1]
        x_to_all = cdist(x, states, metric=metrics)[0]
        topk = np.argpartition(1/(x_to_all + 1e-9), -k)[-k:]
        tok1k_idxs.append(topk)
        topk_distances.append(x_to_all[topk].reshape(1,-1))

        # current_state_idx_in_dataset = tomix_idxs[i]
    topk_distances = np.concatenate(topk_distances, 0)


    distance_infos = {
        "minumum_distances": topk_distances,
        "minumum_distances_idxs": np.array(tok1k_idxs),
        "observation_idxs": observation_idxs,
        "next_observation_idxs":"observation_idxs+1即可",
        "metric": metrics,
        "do_not_select_me": do_not_select_me,
        "metainfo":{
            "what":"what",
            "mixin_idxs":mixin_idxs,
        },

    }

    with open(f"{distance_save_path}/distances_{env}-{dataset}-{ratio:.2f}-v2.pkl", "wb") as f:
        pkl.dump(distance_infos, f)
    return




def generate_clusters_kitchen(name,  metrics = "canberra", vis = True, max_iter = 100): 
    
    distance_save_path = f"dataset_infos"
    if not os.path.exists(distance_save_path):
        os.makedirs(distance_save_path)



    env_obj = load_environment(name)
    origional_trajs, origional_data = sequence_dataset_mix_kitchen(env_obj) 

    # do_not_select_me_expert = []
    # do_not_select_me_origional = []

    origional_states_in_trajs = 0
    for traj in origional_trajs: origional_states_in_trajs += len(traj['observations'])

    with open(f"value_of_states/{name}.pkl", "rb") as f:
        value_of_states = pkl.load(f)

    origional_states = origional_data['observations']
    origional_vs = np.array(value_of_states)

    assert len(origional_vs) == len(origional_states)
    
    if len(origional_states) != origional_states_in_trajs:
        print("Warning: fount incomplete trajectory. Dropping the last one.")
        origional_states = origional_states[:origional_states_in_trajs]
        origional_vs = origional_vs[:origional_states_in_trajs]


    
    tomix_state = []
    tomix_vs = []


    do_not_select_me = [-1]
    for traj in origional_trajs:
        do_not_select_me.append(do_not_select_me[-1] + len(traj['observations']))

    do_not_select_me = do_not_select_me[1:]


    nums_clusters = int(np.sqrt( len(origional_states)  ))

    cluster = MiniBatchKMeans(n_clusters=nums_clusters, batch_size=10240, max_iter=max_iter, verbose=1)
    results = cluster.fit_predict(origional_states)
    
    idxs = list(range(len(results)))

    num_samples_of_clusters = [0 for _ in range(nums_clusters)]
    samples_per_cluster_idx = [[] for _ in range(nums_clusters)]

    num_samples_of_clusters_pure = [0 for _ in range(nums_clusters + 1)]
    samples_per_cluster_idx_pure = [[] for _ in range(nums_clusters + 1)]



    idx = 0
    for c in tqdm(results, desc = "Generating all index."): 
        samples_per_cluster_idx[c].append(idx)
        num_samples_of_clusters[c] += 1
        idx += 1

    results_pure = copy.deepcopy(results)
    results_pure[do_not_select_me] = nums_clusters  
    idx = 0
    for c in tqdm(results, desc = "Generating pure index."): 
        samples_per_cluster_idx_pure[c].append(idx)
        num_samples_of_clusters_pure[c] += 1
        idx += 1


    for idx, samples_in_c in enumerate(samples_per_cluster_idx):
        samples_per_cluster_idx[idx] = np.array(samples_in_c)
    
    for idx, samples_in_c in enumerate(samples_per_cluster_idx_pure):
        samples_per_cluster_idx_pure[idx] = np.array(samples_in_c)

    
        # samples_in_c_pure = [idx for idx in samples_in_c if idx not in do_not_select_me]
        # samples_per_cluster_idx_pure[idx] = np.array(samples_in_c_pure)

    # similarity_in_cluster = []
    # for c in tqdm(range(nums_clusters), desc="Calculating similarity_in_cluster"):
    #     sample_idxs = samples_per_cluster_idx[c]
    #     states_of_c = mixed_states[sample_idxs]
    #     distances = cdist(states_of_c, states_of_c, metric= metrics )
    #     similarity_in_cluster.append(distances)




    cluster_representation_centro = cluster.cluster_centers_     
    similarity_of_clusters = cdist(cluster_representation_centro, cluster_representation_centro, metric= metrics )
    # similarity_sample_to_clusters = cdist(tomix_state, cluster_representation_centro, metric= metrics )

    

        
    cluster_metadata = {
        'cluster_of_samples': results,
        'num_actual_mixed_states': "invalid",
        
        'num_trajectories_to_mixin': "Invalid",
        'origional_states': origional_states,
        'origional_vs': origional_vs,
        'tomix_state': "Invalid",
        'tomix_state_idxs_per_cluster':"Invalid", 
        'samples_per_cluster_idx_pure': samples_per_cluster_idx_pure,
        'num_samples_of_clusters_pure': num_samples_of_clusters_pure,
        'tomix_state_idxs': "Invalid",
        'tomix_state_count': len(tomix_state),
        'tomix_vs': tomix_vs,
        'positive_hash_table': "Please Compute With Run. File too large.",
        'cluster_representation_centro': cluster_representation_centro,
        'nums_clusters': nums_clusters,
        'num_mixed_samples': len(origional_trajs),
        'num_origional_states': len(origional_states),
        'ratio': "Invalid",
        'num_samples_of_clusters': num_samples_of_clusters,
        'samples_per_cluster_idx': samples_per_cluster_idx,
        'similarity_in_cluster': "Please Compute With Run. File too large.", # similarity_in_cluster,
        'similarity_of_clusters': similarity_of_clusters,
        'similarity_sample_to_clusters': "Please Compute With Run. File too large.", # similarity_sample_to_clusters,
        'mixed_do_not_select_me': do_not_select_me,
        'mixed_states': origional_states,
        'mixed_vs': origional_vs,
        'mixed_trajectories': origional_trajs,
        # 'metric': metrics, 
        'generator': "generate_clusters", 
    }


    with open(f"{distance_save_path}/cluster_infos_{name}.pkl", "wb") as f:
        pkl.dump(cluster_metadata, f)
    if vis:
        if len(idxs)//5000 >= 2:
            idxs_to_vis = idxs[::len(results)//5000]
        else:
            idxs_to_vis = idxs
        states_to_vis = origional_states[idxs_to_vis]
        vs_to_vis = origional_vs[idxs_to_vis]
        labels_to_vis = results[idxs_to_vis]
        plt.clf()
        plt.cla()
        cmap = plt.cm.jet


        
        norm = Normalize(vmin=min(vs_to_vis), vmax=max(vs_to_vis))
        manhattan_distances = cdist(states_to_vis, states_to_vis, metric= metrics )#"euclidean")

        umap_reducer = umap.UMAP(n_components=2, metric='precomputed')
        embedding_all = umap_reducer.fit_transform(manhattan_distances)

        
        plt.clf()
        plt.cla()
        plt.figure()
        data = embedding_all
        labels = vs_to_vis
        plt.scatter(data[:, 0], data[:, 1], cmap='viridis',  s =2, c=cmap(norm(labels)))
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='Value')
        plt.savefig(f"{distance_save_path}/{name}_2d_value.png", dpi = 600)
    return




def generate_clusters(env, dataset, ratio = 0.1, nums_clusters = 1000,  metrics = "canberra", vis = False, max_iter = 100):  # 只保留1000个最近的state
    
    distance_save_path = f"dataset_infos"
    if not os.path.exists(distance_save_path):
        os.makedirs(distance_save_path)

        
    with open(f"{distance_save_path}/dataset_info_{env}-expert-v2.pkl", "rb") as f:
        expert_infos = pkl.load(f)
        expert_dataset = expert_infos['dataset']
    with open(f"{distance_save_path}/dataset_info_{env}-{dataset}-v2.pkl", "rb") as f:
        origional_infos = pkl.load(f)
        plain_datset = origional_infos['dataset']
    high_to_low = expert_infos['high_to_low']

    env_obj = load_environment(f"{env}-{dataset}-v2")
    origional_trajs, origional_data = sequence_dataset_mix(env_obj) # 这里出现了差池，origional_trajs总长度比origionaldata中的state少3个


    # do_not_select_me_expert = []
    # do_not_select_me_origional = []

    origional_states_in_trajs = 0
    for traj in origional_trajs: origional_states_in_trajs += len(traj['observations'])



    origional_states = plain_datset['observations']
    origional_vs = origional_infos['vlaues_of_states']

    assert len(origional_vs) == len(origional_states)
    
    if len(origional_states) != origional_states_in_trajs:
        print("Warning: fount incomplete trajectory. Dropping the last one.")
        origional_states = origional_states[:origional_states_in_trajs]
        origional_vs = origional_vs[:origional_states_in_trajs]




    nums_to_mixin =  int(origional_states_in_trajs *ratio )
    mixin_idxs = []
    mixin_lengths = []
    for idx in expert_infos['high_to_low']:
        length = len(expert_infos['trajs'][idx]['observations'])
        mixin_idxs.append(idx)
        mixin_lengths.append(length)
        origional_trajs.append( expert_infos['trajs'][idx]  )
        if np.array(mixin_lengths).sum() >= nums_to_mixin:
            print(f"Added {len(mixin_lengths)} trajs to {env}-{dataset}, {np.array(mixin_lengths).sum()} tris in total ({nums_to_mixin} = {len(plain_datset['observations'])} *{ratio} required.)")
            break
    
    

    tomix_state = []
    tomix_vs = []
    observation_idxs = []
    # next_states = []

    for i, idx in enumerate(mixin_idxs):
        tomix_state_in_traj = list(expert_infos['trajs'][idx]['observations']) 
        trajectory_start_idx, trajectory_end_idx = expert_infos['trajs'][idx]['start'],expert_infos['trajs'][idx]['end']
        if trajectory_end_idx + 1 == trajectory_end_idx + len(tomix_state_in_traj):
            print("wtf2")
        # assert trajectory_end_idx + 1 == trajectory_end_idx + len(tomix_state_in_traj)
        tomix_vs_idx = expert_infos['vlaues_of_states'][trajectory_start_idx: trajectory_end_idx + 1  ] 
        if len(tomix_state_in_traj) != len(tomix_vs_idx):
            assert False
        tomix_state += tomix_state_in_traj 
        tomix_vs += tomix_vs_idx           
        observation_idxs += list(range(trajectory_start_idx, trajectory_end_idx + 1))  

    tomix_state = np.array(tomix_state)
    tomix_vs = np.array(tomix_vs)

    mixed_states_idxs = np.array(observation_idxs)  # un-used

    
    


    mixed_states = np.concatenate((origional_states, tomix_state), 0)
    mixed_vs = np.concatenate((origional_vs, tomix_vs), 0)
    # mixed_do_not_select_me = [ do_not_select_me_origional, len(mixed_states) - 1  ]
    mixed_trajectories = origional_trajs
    tomix_state_idxs = list(range( len(origional_states), len(mixed_states)  ))
    do_not_select_me = [-1]
    for traj in mixed_trajectories:
        do_not_select_me.append(do_not_select_me[-1] + len(traj['observations']))

    do_not_select_me = do_not_select_me[1:]


    nums_clusters = int(np.sqrt( len(mixed_states)  ))

    cluster = MiniBatchKMeans(n_clusters=nums_clusters, batch_size=10240, max_iter=max_iter, verbose=1)
    results = cluster.fit_predict(mixed_states)
    
    idxs = list(range(len(results)))

    num_samples_of_clusters = [0 for _ in range(nums_clusters)]
    samples_per_cluster_idx = [[] for _ in range(nums_clusters)]

    num_samples_of_clusters_pure = [0 for _ in range(nums_clusters + 1)]
    samples_per_cluster_idx_pure = [[] for _ in range(nums_clusters + 1)]

    tomix_expert_state_idxs_per_cluster = [[] for _ in range(nums_clusters)]


    num_samples_of_clusters_do_not_select = [0 for _ in range(nums_clusters + 1)]
    samples_per_cluster_idx_do_not_select = [[] for _ in range(nums_clusters + 1)]

    idx = 0
    for c in tqdm(results, desc = "Generating all index."): 
        samples_per_cluster_idx[c].append(idx)
        num_samples_of_clusters[c] += 1
        idx += 1

    results_pure = copy.deepcopy(results)
    results_pure[do_not_select_me] = nums_clusters  
    idx = 0
    for c in tqdm(results, desc = "Generating pure index."): 
        samples_per_cluster_idx_pure[c].append(idx)
        num_samples_of_clusters_pure[c] += 1
        idx += 1

    for idx, c in enumerate(results[len(origional_states):  ]): 
        print(f"Processing:{100*idx/len(tomix_state):.4f} %")
        tomix_expert_state_idxs_per_cluster[c].append(idx)

    for idx, samples_in_c in enumerate(samples_per_cluster_idx):
        samples_per_cluster_idx[idx] = np.array(samples_in_c)
    
    for idx, samples_in_c in enumerate(samples_per_cluster_idx_pure):
        samples_per_cluster_idx_pure[idx] = np.array(samples_in_c)

    
        # samples_in_c_pure = [idx for idx in samples_in_c if idx not in do_not_select_me]
        # samples_per_cluster_idx_pure[idx] = np.array(samples_in_c_pure)

    # similarity_in_cluster = []
    # for c in tqdm(range(nums_clusters), desc="Calculating similarity_in_cluster"):
    #     sample_idxs = samples_per_cluster_idx[c]
    #     states_of_c = mixed_states[sample_idxs]
    #     distances = cdist(states_of_c, states_of_c, metric= metrics )
    #     similarity_in_cluster.append(distances)




    cluster_representation_centro = cluster.cluster_centers_     
    similarity_of_clusters = cdist(cluster_representation_centro, cluster_representation_centro, metric= metrics )
    # similarity_sample_to_clusters = cdist(tomix_state, cluster_representation_centro, metric= metrics )

    if vis:
        idxs_to_vis = idxs[::len(results)//5000]
        states_to_vis = mixed_states[idxs_to_vis]
        vs_to_vis = mixed_vs[idxs_to_vis]
        labels_to_vis = results[idxs_to_vis]
        plt.clf()
        plt.cla()
        cmap = plt.cm.jet


        
        norm = Normalize(vmin=min(vs_to_vis), vmax=max(vs_to_vis))
        manhattan_distances = cdist(states_to_vis, states_to_vis, metric= metrics )#"euclidean")

        umap_reducer = umap.UMAP(n_components=2, metric='precomputed')
        embedding_all = umap_reducer.fit_transform(manhattan_distances)

        
        plt.clf()
        plt.cla()
        plt.figure()
        data = embedding_all
        labels = vs_to_vis
        plt.scatter(data[:, 0], data[:, 1], cmap='viridis',  s =2, c=cmap(norm(labels)))
        base_filename = f"{env}-{dataset}-{ratio:.2f}"
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='Value')
        plt.savefig(f"{base_filename}_2d_value.png", dpi = 600)

        plt.clf()
        plt.cla()
        plt.figure()
        data = embedding_all
        plt.scatter(data[:, 0], data[:, 1], cmap='viridis', s = 2, c = labels_to_vis)
        base_filename = f"{env}-{dataset}-{ratio:.2f}"
        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), label='ClusterID')
        plt.savefig(f"{base_filename}_2d_vlabel.png", dpi = 600)
    # build idx map
    
    # idx_in_cluster_map_cache = []
    # for c in tqdm(list(range(nums_clusters)), desc = "Building idx - idx_in_cluster map cache"):
    #     map_c = {}
    #     map_c['clusterid'] = c
    #     map_c['numbers'] = len(samples_per_cluster_idx[c])
    #     for index, sample_idx in enumerate(samples_per_cluster_idx[c]):
    #         map_c[sample_idx] = index
    #     idx_in_cluster_map_cache.append(map_c)
    

    # return
    cluster_metadata = {
        'cluster_of_samples': results,
        'num_actual_mixed_states': len(tomix_state),
        
        'num_trajectories_to_mixin': nums_to_mixin,
        'origional_states': origional_states,
        'origional_vs': origional_vs,
        'tomix_state': tomix_state,
        'tomix_state_idxs_per_cluster':tomix_expert_state_idxs_per_cluster, 
        'samples_per_cluster_idx_pure': samples_per_cluster_idx_pure,
        'num_samples_of_clusters_pure': num_samples_of_clusters_pure,
        'tomix_state_idxs': tomix_state_idxs,
        'tomix_state_count': len(tomix_state),
        'tomix_vs': tomix_vs,
        'positive_hash_table': "Please Compute With Run. File too large.",
        'cluster_representation_centro': cluster_representation_centro,
        'nums_clusters': nums_clusters,
        'num_mixed_samples': len(mixed_states),
        'num_origional_states': len(origional_states),
        'ratio': ratio,
        'num_samples_of_clusters': num_samples_of_clusters,
        'samples_per_cluster_idx': samples_per_cluster_idx,
        'similarity_in_cluster': "Please Compute With Run. File too large.", # similarity_in_cluster,
        'similarity_of_clusters': similarity_of_clusters,
        'similarity_sample_to_clusters': "Please Compute With Run. File too large.", # similarity_sample_to_clusters,
        'mixed_do_not_select_me': do_not_select_me,
        'mixed_states': mixed_states,
        'mixed_vs': mixed_vs,
        'mixed_trajectories': mixed_trajectories,
        # 'metric': metrics, 
        'generator': "generate_clusters", 
    }


    with open(f"{distance_save_path}/cluster_infos_{env}-{dataset}-{ratio:.2f}-v2.pkl", "wb") as f:
        pkl.dump(cluster_metadata, f)
    return


def predownload(env_name):
    env = load_environment(env_name)
    trajs, _ = sequence_dataset_mix(env)

    return


meta_infos = {
    "maze2d": {
        'tasks': ['maze2d'],
        'datasets':["umaze", "medium", "large","umaze-dense", "medium-dense", "large-dense"],
        "versions":["v1"],
        "sparse": "Sparse except dense",
        "observshape": ""
    },
    "antmaze": {
        'tasks':['antmaze'],
        'datasets':["umaze", "umaze-diverse", "medium-diverse","medium-play", "large-diverse", "large-play"],
        "versions":["v0"],
        "sparse": "Sparse"
    },
    "adroit": {
        'tasks': ['pen', 'hammer', 'door', 'relocate'],
        'datasets':['human', 'cloned', 'expert'],
        "versions":["v1"],
        "sparse": "Dense"
    },
    "gym": {
        'tasks': ['halfcheetah', 'walker2d', 'hopper', 'ant'],
        'datasets':['random', 'medium', 'expert', 'medium-expert', 'medium-replay'],
        "versions":["v2"],
        "sparse": "Dense"
    },
    "frankakitchen": {
        'tasks': ['kitchen'],
        'datasets':['complete', 'partial', 'mixed'],
        "versions":["v0"],
        "sparse": "Dense"
    },
}


if __name__ == "__main__":

    import sys
    import os
    current_file_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(current_file_path)
    os.chdir(parent_directory)
    Configs.logger = MyLogger("~/cache/log.log")

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--dataset", type=str, default="medium")
    parser.add_argument("--env", type=str, default="frankakitchen")
    parser.add_argument("--ratio", type=float, default=0.01)
    parser.add_argument("--exp_dataset", type=str, default="expert")
    parser.add_argument("--expert_ratio", type=float, default=0.1)
    args = parser.parse_args()
    # gen_value_of_states(args.dataset)
    # generate_expert_info()
    # for ratio in [0.01, 0.1, 0.2, 0.3]:
    #     generate_mix_datasets(ratio = 0.01)
    #     vis_mixed_rewards(ratio)
    # assert_dataset_infos()
    envs = ["walker2d", "hopper", "halfcheetah"]
    datasets = ["random", "medium", "medium-replay", "expert"]
    ratios = [0.01, 0.05, 0.1, 0.2, 0.3]
    # for dataset in datasets:
    #     generate_dataset_infos(args.env, dataset)
    # for env in envs:
    env = args.env
    dataset = args.dataset

    generate_clusters_kitchen("kitchen-mixed-v0")
    sys.exit(0)
    
    block = meta_infos[env]
    # generate_clusters_kitchen("kitchen-mixed-v0")
    for task in block['tasks']:
        for dataset in block['datasets']:
            version = block['versions'][0]
            env_name = f"{task}-{dataset}-{version}"
            # render_dataset(env, env_name,)
            # gen_value_of_states(env_name)
            # generate_dataset_infos(env_name, dataset)
            # gen_value_of_states(env_name)
            generate_clusters_kitchen(env_name)
            # generate_clusters()
            # sys.exit(1)
    # predownload("ant-expert-v2")
    
    # gen_value_of_states("ant-expert-v2")
    # generate_dataset_infos("ant", "expert")
    # for dataset in datasets[:-2]:
    # for ratio in ratios:
    #     # visual_embeddings_2d(env, dataset, ratio)
    #     # for metrics in ['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',        'kulczynski1', 'mahalanobis', 'matching', 'minkowski',        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',        'sokalsneath', 'sqeuclidean', 'yule']:
    #     visual_precomputed(env, dataset, ratio,metrics = "canberra")
            # visual_embeddings(env, dataset, ratio)
    # generate_dataset_infos(env,"expert")
    # for env in envs:
    #     for dataset in datasets:
    #         generate_dataset_infos(env,dataset)
    # generate_dataset_infos(env, "expert")
    # generate_distances(env, dataset = "expert", ratio=args.ratio, k = 1000)
    # for ratio in ratios:
    #     for dataset in datasets[:-1]:
    #         generate_clusters(env, dataset = dataset, ratio=ratio, nums_clusters = 1000, max_iter = 10000)
            # sys.exit(0)

    # generate_clusters("ant", dataset = "expert", ratio=0.01, nums_clusters = -1, max_iter = 100000, vis = False)
    # for env in envs:
    #     generate_distances(env, dataset = "expert", ratio=0.3, k = 1000)