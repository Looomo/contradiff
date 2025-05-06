# from queue import Queue
from collections import namedtuple
import threading
import numpy as np
import torch
import pdb
import pickle as pkl
from tqdm import tqdm

from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset, sequence_dataset_plain
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer
from scipy.spatial.distance import cdist

Batch = namedtuple('Batch', 'trajectories conditions')
TransCondBatch = namedtuple('TransCondBatch', 'trajectories conditions history')
TransCondBatch_plab3a = namedtuple('TransCondBatch_plab3a', 'trajectories conditions history_obs history_act')
ContrastiveBatch = namedtuple('ContrastiveBatch', 'trajectories conditions positives negatives')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')
HistoryContrastiveBatch = namedtuple('HistoryContrastiveBatch', 'trajectories conditions positives negatives history')
Batch_plan9 = namedtuple('Batch_plan9', 'trajectories conditions positives negatives positive_vals negative_vals history')
Batch_plan14bf = namedtuple('Batch_plan14bf', 'trajectories conditions positives negatives positive_vals negative_vals traj_rewards')
Batch_plan14e = namedtuple('Batch_plan14e', 'trajectories conditions rewards')
Batch_plan2 = namedtuple('Batch_plan2', 'trajectories conditions positives negatives positive_vals negative_vals traj_rewards')


from config.locomotion_config import Configs

from tqdm import tqdm
import pickle
import copy
import umap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import inspect
import os

class MixDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.normalizer_str = normalizer
        discount = Configs.contrast_discount
        self.discount = discount
        self.seed = seed
        
        self.termination_penalty = termination_penalty

        self.returns_scale = Configs.returns_scale

        self.env_name = env
        name_metas = self.env_name.split("-")
        self.expert_env_name = f"{name_metas[0]}-{Configs.exp_dataset}-v2"
        # self.mixed_dataset_name = f"{self.env_name.split('-')[0]}-{Configs.exp_dataset}-v2"
        self.env = env = load_environment(env)
        self.env.seed(seed)
        
        
        max_path_length = self.env.max_episode_steps
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding

        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]

        fields = self.load_data()

        self.fields = fields
        self.normalizer = DatasetNormalizer(self.fields, self.normalizer_str, path_lengths=self.fields['path_lengths'])

        self.observation_dim = self.fields.observations.shape[-1]
        self.action_dim = self.fields.actions.shape[-1]
        self.n_episodes = self.fields.n_episodes
        self.path_lengths = self.fields.path_lengths
        self.normalize()

        self.indices = self.make_indices(self.fields.path_lengths, self.horizon)


        self.vis_value_of_states()

    def sort_by_values(self, to_sort, values):
        inds = np.argsort(values)[::-1]
        # if to_sort:
        to_sort = to_sort[inds]
        values = values[inds]
        return to_sort, values
    
    def load_data(self,):
        
        self.expert_env = load_environment(self.expert_env_name)
        self.expert_env.seed(self.seed)

        itr, plain_datset = sequence_dataset_plain(self.env, self.preprocess_fn)
        self.plain_datset = plain_datset
        # with open(  f"mixed_datasets/{self.env_name.replace('-v2', '')}-mix-expert-{Configs.expert_ratio:.2f}-v2.pkl" , "rb"  ) as f:
        #     self.mixed_dataset = pkl.load(f) # dict_keys(['trajs', 'mixed_expert_inds', 'expert_dataset', 'plain_vlaues_of_states'])
        dataset_infos_path = "dataset_infos"
        if hasattr(Configs, "dataset_infos_path" ):
            dataset_infos_path = Configs.dataset_infos_path

        dataset_infos_file_path = os.path.join( dataset_infos_path,  f"dataset_info_{self.expert_env_name}.pkl"  )
        with open( f"./dataset_infos/dataset_info_{self.expert_env_name}.pkl"  , "rb"  ) as f:
            self.expert_dataset = pkl.load(f)
        # self.expert_dataset = self.mixed_dataset['expert_dataset'] # dict_keys(['dataset', 'trajs', 'accumulated_rewards', 'high_to_low', 'vlaues_of_states'])
        
        self.nums_to_mixin =  int(len(self.plain_datset['observations']) *Configs.expert_ratio )
        self.mixin_idxs = []
        self.mixin_lengths = []
        for idx in self.expert_dataset['high_to_low']:
            length = len(self.expert_dataset['trajs'][idx]['observations'])
            self.mixin_idxs.append(idx)
            self.mixin_lengths.append(length)
            itr.append( self.expert_dataset['trajs'][idx]   )
            if np.array(self.mixin_lengths).sum() >= self.nums_to_mixin:
                Configs.logger.log(f"Added {len(self.mixin_lengths)} trajs to {self.env_name}, {np.array(self.mixin_lengths).sum()} tris in total ({self.nums_to_mixin} = {len(self.plain_datset['observations'])} *{Configs.expert_ratio} required.)")
                break

        self.num_trajs_after_mix = len(itr)
        random.shuffle(itr)
        fields = ReplayBuffer(len(itr) + 3000, self.max_path_length, self.termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode,  rewrite="kitchen" in Configs.dataset, skip_items=[ 'start', 'end', 'accumulated_reward'  ])
        fields.finalize()
        value_of_states_path = "value_of_states"
        if hasattr(Configs, "value_of_states_path" ):
            value_of_states_path = Configs.value_of_states_path
        vales_file_path = os.path.join( "./value_of_states",  f"{self.env_name}.pkl"  )
        with open(vales_file_path , "rb") as f:
            vs_orig = pkl.load(f)
        # with open(f"./value_of_states/{self.mixed_dataset_name}.pkl", "rb") as f:
        #     vs_expert = pkl.load(f)  
        tomix_state = []
        tomix_vs = []
        mixed_state_idxs = []
        for i, idx in enumerate(self.mixin_idxs):
            tomix_state_idx = list(self.expert_dataset['trajs'][idx]['observations'])
            trajectory_start_idx, trajectory_end_idx = self.expert_dataset['trajs'][idx]['start'],self.expert_dataset['trajs'][idx]['end']
            tomix_vs_idx = self.expert_dataset['vlaues_of_states'][trajectory_start_idx:trajectory_end_idx + 1]  # Fixed
            mixed_state_idxs += list(range(trajectory_start_idx,trajectory_end_idx + 1))
            if len(tomix_state_idx) != len(tomix_vs_idx):
                assert False
            tomix_state += tomix_state_idx
            tomix_vs += tomix_vs_idx

        
        self.states_orig = list(plain_datset['observations'])
        self.states_to_mix = tomix_state
        self.mixed_state = tomix_state + list(plain_datset['observations'])

        self.vs_orig = vs_orig
        self.tomix_vs = tomix_vs
        self.mixed_vs = tomix_vs + vs_orig
        self.mixed_state_idxs = mixed_state_idxs

        self.mixed_vs = np.array(self.mixed_vs)

        self.num_states_after_mix = len(self.mixed_state)

        assert len(self.tomix_vs) == len(self.states_to_mix)
        self.num_actual_mixed_states = len(self.tomix_vs)

        # idxs = list(range(len(self.mixed_state)))
        # assert len(self.mixed_state) == len(self.mixed_vs)

        # will be normed in self.norm---

        return fields


    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

        self.mixed_state_normed = self.normalizer(np.array(self.mixed_state), "observations")

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices
    
    def vis_value_of_states(self, ):
        savepath = os.path.join(Configs.savepath, "vis")
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        plt.cla()
        plt.clf()
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)

        
        idxs_1, sorted_values = self.sort_by_values(np.array(list(range(len(self.mixed_vs)))), np.array(self.mixed_vs) )
        sorted_values = sorted_values[::sorted_values.shape[0]//100]
        axes[0].bar(list(range(len(sorted_values))), sorted_values, color='blue')
        axes[0].set_title(f'Mixed {self.env_name}-{Configs.exp_dataset}-{Configs.expert_ratio:.2f}')

        idxs_1, sorted_values = self.sort_by_values(np.array(list(range(len(self.vs_orig)))), np.array(self.vs_orig) )
        sorted_values = sorted_values[::sorted_values.shape[0]//100]
        axes[1].bar(list(range(len(sorted_values))), sorted_values, color='blue')
        axes[1].set_title(f'Origional {self.env_name}-{Configs.exp_dataset}-{Configs.expert_ratio:.2f}')

        infos = f'''
            mixed states: {self.num_actual_mixed_states}
            mixed trajs: {self.nums_to_mixin}
            total after mix: {len(self.mixed_vs)} states, {self.num_trajs_after_mix} trajs
        '''
        axes[1].text(0.4, 0.8, infos, fontsize=12, ha='center', va='center', transform=axes[1].transAxes)


        plt.savefig( os.path.join(  savepath,   f'Values-{self.env_name}-{Configs.exp_dataset}-{Configs.expert_ratio:.2f}.png'  ) , dpi = 300)
        return 

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        
        return {0: observations[0]} if Configs.with_cond else {}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]

        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch
    

class MixDataset_plan1_diffuser(MixDataset):
    def __init__(self, env='hopper-medium-replay', horizon=64, normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000, max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None):
        super().__init__(env, horizon, normalizer, preprocess_fns, max_path_length, max_n_episodes, termination_penalty, use_padding, seed)

class MixDataset_plan1_hard(MixDataset):
    def __init__(self, env='hopper-medium-replay', horizon=64, normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000, max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None):
        super().__init__(env, horizon, normalizer, preprocess_fns, max_path_length, max_n_episodes, termination_penalty, use_padding, seed)

        self.make_contrast_samples()
        self.placeholder = np.array(128, dtype=np.float32)
    
    def make_contrast_samples(self,):
        if Configs.upperbound > 0:
            posi_ratio = Configs.upperbound
        else:
            posi_ratio = Configs.expert_ratio
            Configs.logger.log(f"!!!!!! Setting posi_ratio to expert_ratio ({posi_ratio}), as its set to automatic detected (Configs.upperbound = {Configs.upperbound}).")
        nega_ratio = Configs.lowerbound
        posi_nums = int(posi_ratio*self.num_states_after_mix)
        nega_nums = int(nega_ratio*self.num_states_after_mix)

        self.positive_nums = posi_nums
        self.negative_nums = nega_nums

        states_idxs = list(range(  self.num_states_after_mix  ))

        # high to low sorted
        sorted_idx, sorted_value = self.sort_by_values( np.array(states_idxs), np.array( self.mixed_vs)  )

        posi_idxs = sorted_idx[:posi_nums]
        nega_idxs = sorted_idx[-nega_nums:]

        random.shuffle(posi_idxs)
        random.shuffle(nega_idxs)

        self.positive_state_idxs = posi_idxs
        self.negative_state_idxs = nega_idxs

        if Configs.save_contrast_splits:
            self.positive_state_idxs = posi_idxs
            self.negative_state_idxs = nega_idxs

            posi_rewards = self.mixed_vs[posi_idxs]
            posi_states = self.mixed_state_normed[posi_idxs]

            nega_rewards = self.mixed_vs[nega_idxs]
            nega_states = self.mixed_state_normed[nega_idxs]

            data = {
                "posi_rewards": posi_rewards,
                "posi_states": posi_states,
                "nega_rewards": nega_rewards,
                "nega_states":nega_states
            }
            
            with open(f"{Configs.savepath}/data_{Configs.dataset}.pkl", "wb") as f:
                pkl.dump(data, f)

        return

    def get_positive_samples(self, ):

        positive_idxs_idx = (np.random.randint(self.positive_nums, size = Configs.subbatchsize*Configs.horizon))
        positive_idxs = self.positive_state_idxs[positive_idxs_idx]
        positive_states = self.mixed_state_normed[positive_idxs]
        positive_rewards = self.mixed_vs[positive_idxs]
        positive_states = positive_states.reshape( Configs.subbatchsize, Configs.horizon, -1 )
        positive_rewards = positive_rewards.reshape( Configs.subbatchsize, Configs.horizon, -1 )

        return positive_states, positive_rewards
    
    def get_negative_samples(self, ):
        if Configs.lowerbound > 0:
            negative_idxs_idx = (np.random.randint(self.negative_nums, size = Configs.subbatchsize*Configs.horizon  ))
            negative_idxs = self.negative_state_idxs[negative_idxs_idx]
            negative_states = self.mixed_state_normed[negative_idxs]
            negative_rewards = self.mixed_vs[negative_idxs]
            negative_states = negative_states.reshape( Configs.subbatchsize, Configs.horizon, -1 )
            negative_rewards = negative_rewards.reshape( Configs.subbatchsize, Configs.horizon, -1 )
        else:
            negative_states = self.placeholder
            negative_rewards = self.placeholder
        return negative_states, negative_rewards
    

    def __getitem__(self, idx, eps=1e-4):

        path_ind, start,  end = self.indices[idx]
        # history = self.fields.normed_observations[path_ind, start:end]
        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        positive_states, positive_rewards = self.get_positive_samples()
        negative_states, negative_rewards = self.get_negative_samples()

        batch = Batch_plan14bf(trajectories, conditions, positive_states, negative_states, positive_rewards, negative_rewards, traj_rewards = self.placeholder)

        return batch
    

class MixDataset_plan1_hardnce(MixDataset_plan1_hard):
    def __init__(self, env='hopper-medium-replay', horizon=64, normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000, max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None):
        super().__init__(env, horizon, normalizer, preprocess_fns, max_path_length, max_n_episodes, termination_penalty, use_padding, seed)



class MixDataset_plan2_hard(MixDataset_plan1_hard):
    def __init__(self, env='hopper-medium-replay', horizon=64, normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000, max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None):
        super().__init__(env, horizon, normalizer, preprocess_fns, max_path_length, max_n_episodes, termination_penalty, use_padding, seed)
        self.contrast_place_holder = np.zeros((Configs.subbatchsize, Configs.horizon,self.observation_dim)).astype(np.float32)
    def calculate_similarity_in_cluster(self, ):
        nums_clusters = self.cluster_info['nums_clusters']
        samples_per_cluster_idx = self.cluster_info['samples_per_cluster_idx']
        mixed_states = self.cluster_info['mixed_states']
        cluster_representation_centro = self.cluster_info['cluster_representation_centro']
        metrics = Configs.metrics
        similarity_in_cluster = []
        for c in tqdm(range(nums_clusters), desc="Calculating similarity_in_cluster"):
            sample_idxs = samples_per_cluster_idx[c]
            states_of_c = mixed_states[sample_idxs]
            distances = cdist(states_of_c, states_of_c, metric= metrics ).astype(np.float32)
            similarity_in_cluster.append(distances)
        
        # self.similarity_sample_to_clusters = cdist(mixed_states, cluster_representation_centro, metric= metrics ).astype(np.float32)
        self.similarity_in_cluster = similarity_in_cluster
        return
    def build_hash_table(self,):

        nums_clusters = self.cluster_info['nums_clusters']

        
        
        samples_per_cluster_idx_pure = self.cluster_info['samples_per_cluster_idx_pure'][:-1] # 去掉z以后一个 最后一个是被删掉不考虑的
        samples_per_cluster_idx = self.cluster_info['samples_per_cluster_idx']

        idx_in_cluster_map_cache = []
        for c in tqdm(list(range(nums_clusters)), desc = "Building idx - idx_in_cluster map cache"):
            map_c = {}
            map_c['clusterid'] = c
            map_c['numbers'] = len(samples_per_cluster_idx[c])
            for index, sample_idx in enumerate(samples_per_cluster_idx[c]):
                map_c[sample_idx] = index  # 映射为 global idx -> in-cluster idx
            idx_in_cluster_map_cache.append(map_c)
        
        mixed_states = self.cluster_info['mixed_states']
        results = self.cluster_info['cluster_of_samples']



        # 用于记录每个簇内哪些样本不应该被选择
        do_not_select_by_clusters = [ [] for _ in range(nums_clusters)  ]
        for do_not_select in tqdm(self.cluster_info['mixed_do_not_select_me'], desc = "Grouping  do_not_select_me by clusters."):
            c = self.cluster_info['cluster_of_samples'][do_not_select]
            do_not_select_by_clusters[c].append(idx_in_cluster_map_cache[c][do_not_select])
        total_do_not_select = 0
        for c in range(nums_clusters): total_do_not_select+= len(do_not_select_by_clusters[c])
        states_idxs = list(range(len(mixed_states)))
        positive_hash_table = [ [] for _ in states_idxs ]


        for s_idx in tqdm(states_idxs, desc="Building positive hash table"):
            cluster_of_s_idx = results[s_idx]  # 当前样本的cluster id
            
            s_idx_in_c = idx_in_cluster_map_cache[cluster_of_s_idx][s_idx]  # 获取当前样本在cluster内的id

            similarity_to_samples_in_cluster = self.similarity_in_cluster[cluster_of_s_idx][s_idx_in_c] # 获取当前样本与簇内其他样本的相似度。在这里应该把不考虑的那些去掉(置为无穷大，表示不会被选到)
            do_not_select_idx_in_cluster = do_not_select_by_clusters[cluster_of_s_idx]
            similarity_to_samples_in_cluster[do_not_select_idx_in_cluster] = 1e9 # 与do_not_select_me的距离设为无穷大

            sample_idx_in_c = samples_per_cluster_idx[cluster_of_s_idx] # 同一个cluster内的所有样本的id
            k = min(self.cluster_info['nums_clusters'], len(sample_idx_in_c) - len(do_not_select_idx_in_cluster) )
            topk = np.argpartition(1/(similarity_to_samples_in_cluster + 1e-5), -k)[-k:]  # 距离最小的k个样本
            
            candidates = sample_idx_in_c[topk]  # 取出top1000的样本的global id
            
            values = self.mixed_vs[candidates + 1]  # 这里按照真实id+1取value，也就是取可到达的样本的value
            k_values = min(int(self.cluster_info['nums_clusters']*Configs.expert_ratio), len(values))
            topk_values = np.argpartition(values, -k_values)[-k_values:]   # 取下一条value最高的100个样本
            positive_hash_table[s_idx] = candidates[topk_values]
        
        del self.similarity_in_cluster
        self.positive_hash_table = positive_hash_table
        
    def load_data(self,):
        
        with open(f"dataset_infos/cluster_infos_{self.env_name.replace('-v2', f'-{Configs.expert_ratio:.2f}-v2')}.pkl", "rb") as f:
            self.cluster_info = pkl.load(f) 
        
        self.calculate_similarity_in_cluster()

        itr = self.cluster_info['mixed_trajectories']
        self.num_trajs_after_mix = len(itr)
        trajectory_index = list(range(self.num_trajs_after_mix))
        random.shuffle(trajectory_index)
        # itr = [self.cluster_info['mixed_trajectories'][idx] for idx in trajectory_index]
        self.trajectory_index_map = trajectory_index # 原始 -> 新的
        fields = ReplayBuffer(len(itr) + 3000, self.max_path_length, self.termination_penalty)

        map_back_lenghts = [-1 for _ in range(len(itr))]
        for i, index in enumerate(trajectory_index):
            fields.add_path(itr[index],  rewrite="kitchen" in Configs.dataset, skip_items=[ 'start', 'end', 'accumulated_reward'  ])
            map_back_lenghts[index] = len(itr[index]['observations'])
        fields.finalize()

        
        self.map_back_lenghts = map_back_lenghts
        self.trajectory_index = trajectory_index

        for index, length in enumerate(map_back_lenghts):
            assert length == len(self.cluster_info['mixed_trajectories'][index]['observations'])

        self.num_states_after_mix = len(self.cluster_info['mixed_states'])
        self.mixed_vs = self.cluster_info['mixed_vs']
        self.mixed_state = self.cluster_info['mixed_states']

        self.vs_orig = self.cluster_info['origional_vs']
        self.nums_to_mixin = self.cluster_info['num_trajectories_to_mixin']
        self.num_trajs_after_mix = len(itr)

        self.num_actual_mixed_states = self.cluster_info['num_actual_mixed_states']
        self.mixed_do_not_select_me = self.cluster_info['mixed_do_not_select_me']
        
        # self.hashtable = self.cluster_info['positive_hash_table']


        return fields
    

    def make_contrast_samples(self,):
        if Configs.lowerbound > 0:
            nega_ratio = Configs.lowerbound
            nega_nums = int(nega_ratio*self.num_states_after_mix)
            self.positive_nums = self.num_actual_mixed_states
            self.negative_nums = nega_nums
            states_idxs = list(range(  self.num_states_after_mix  ))
            # high to low sorted
            sorted_idx, sorted_value = self.sort_by_values( np.array(states_idxs), np.array( self.mixed_vs)  )
            nega_idxs = sorted_idx[-self.negative_nums:]
            random.shuffle(nega_idxs)
            self.negative_state_idxs = nega_idxs
        else:
            self.negative_state_idxs = None
        # 需要控制内存用量。目前33G -> 93G, 60Gx12 = 720g > Max Mem = 500G
        self.build_hash_table()
        return
    
    # 只需要重载正样本的构建方式
    def get_positive_samples(self, path_ind, start,  end ):

        true_traj_idx = self.trajectory_index[path_ind]
        start_state_idx =  sum(self.map_back_lenghts[:true_traj_idx]) + start

        if start_state_idx in self.mixed_do_not_select_me:
            print("Found----------------------")
            return self.contrast_place_holder, self.contrast_place_holder, np.array([0])

        positive_samples = []
        positive_values = []
        for state_idxs_for_traj in range(Configs.subbatchsize): # positive_path:
            idxs = [start_state_idx]   
            for idx in range(Configs.horizon-1): # state_idxs_for_traj:
                candidates = len(self.positive_hash_table[idxs[-1]])
                idxs.append(  self.positive_hash_table[idxs[-1]][ np.random.randint(candidates)]  + 1  )
            traj = self.mixed_state_normed[idxs]
            vals = self.mixed_vs[idxs]
            positive_samples.append(traj)
            positive_values.append(vals)

        
        positive_states = np.array(positive_samples).reshape( Configs.subbatchsize, Configs.horizon, -1 )
        positive_rewards = np.array(positive_values).reshape( Configs.subbatchsize, Configs.horizon, -1 )

        # return self.contrast_place_holder, self.contrast_place_holder, np.array([0])
        return positive_states, positive_rewards
    

    def __getitem__(self, idx, eps=1e-4):

        path_ind, start,  end = self.indices[idx]
        # history = self.fields.normed_observations[path_ind, start:end]
        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([actions, observations], axis=-1)

        positive_states, positive_rewards = self.get_positive_samples(path_ind, start,  end)
        negative_states, negative_rewards = self.get_negative_samples()

        batch = Batch_plan14bf(trajectories, conditions, positive_states, negative_states, positive_rewards, negative_rewards, traj_rewards = self.placeholder)

        return batch



class MixDataset_plan2_hardnce(MixDataset_plan2_hard):
    def __init__(self, env='hopper-medium-replay', horizon=64, normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000, max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None):
        super().__init__(env, horizon, normalizer, preprocess_fns, max_path_length, max_n_episodes, termination_penalty, use_padding, seed)


class MixDataset_plan2_nomix(MixDataset_plan2_hard):
    def __init__(self, env='hopper-medium-replay', horizon=64, normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000, max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None):
        super().__init__(env, horizon, normalizer, preprocess_fns, max_path_length, max_n_episodes, termination_penalty, use_padding, seed)
    
    def vis_value_of_states(self,):
        return

    def load_data(self,):
        
        with open(f"dataset_infos/cluster_infos_{self.env_name}.pkl", "rb") as f:
            self.cluster_info = pkl.load(f) 
        
        self.calculate_similarity_in_cluster()
        # itr, plain_datset = sequence_dataset_plain(self.env, None)
        itr = self.cluster_info['mixed_trajectories']
        self.max_path_length = np.max([  len(traj['observations']) for traj in itr ] )
        self.num_trajs_after_mix = len(itr)
        trajectory_index = list(range(self.num_trajs_after_mix))
        random.shuffle(trajectory_index)
        # itr = [self.cluster_info['mixed_trajectories'][idx] for idx in trajectory_index]
        self.trajectory_index_map = trajectory_index 
        fields = ReplayBuffer(len(itr) + 3000, self.max_path_length, self.termination_penalty)

        map_back_lenghts = [-1 for _ in range(len(itr))]
        for i, index in enumerate(trajectory_index):
            fields.add_path(itr[index],  rewrite="kitchen" in Configs.dataset, skip_items=[ 'start', 'end', 'accumulated_reward'  ])
            map_back_lenghts[index] = len(itr[index]['observations'])
        fields.finalize()

        
        self.map_back_lenghts = map_back_lenghts
        self.trajectory_index = trajectory_index

        for index, length in enumerate(map_back_lenghts):
            assert length == len(self.cluster_info['mixed_trajectories'][index]['observations'])

        self.mixed_do_not_select_me = self.cluster_info['mixed_do_not_select_me']
        

        self.mixed_state = list(  self.cluster_info['mixed_states'] )
        self.mixed_vs = np.array(self.cluster_info['mixed_vs'])
        self.num_states = len(self.mixed_state)
        
        return fields
    
    def make_contrast_samples(self,):
        if Configs.lowerbound > 0:
            nega_ratio = Configs.lowerbound
            nega_nums = int(nega_ratio*self.num_states)
            # self.positive_nums = self.num_states # unyused
            self.negative_nums = nega_nums
            states_idxs = list(range(  self.num_states  ))
            # high to low sorted
            sorted_idx, sorted_value = self.sort_by_values( np.array(states_idxs), np.array( self.mixed_vs)  )
            nega_idxs = sorted_idx[-self.negative_nums:]
            random.shuffle(nega_idxs)
            self.negative_state_idxs = nega_idxs
        else:
            self.negative_state_idxs = None

        self.build_hash_table()
        return
    
class MixDataset_plan1_nomix(MixDataset_plan1_hard):
    def __init__(self, env='hopper-medium-replay', horizon=64, normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000, max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None):
        super().__init__(env, horizon, normalizer, preprocess_fns, max_path_length, max_n_episodes, termination_penalty, use_padding, seed)
    

    def vis_value_of_states(self,):
        return
    def load_data(self):
        
        itr, plain_datset = sequence_dataset_plain(self.env, None)
        self.max_path_length = np.max([  len(traj['observations']) for traj in itr ] )
        self.plain_datset = plain_datset

        self.num_trajs_after_mix = len(itr)
        random.shuffle(itr)
        fields = ReplayBuffer(len(itr) + 10, self.max_path_length, self.termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode,  rewrite=False, skip_items=[ 'start', 'end', 'accumulated_reward'  ])
        fields.finalize()

        self.states = plain_datset['observations']
        # self.rewards = list(plain_datset['rewards'])
        self.num_states = len(self.states)
        with open(f"value_of_states/{self.env_name}.pkl", "rb") as f:
            vlaues_of_states = pkl.load(f)

        self.vlaues_of_states = np.array(vlaues_of_states)
        return fields
    
    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

        self.states_normed = self.normalizer(np.array(self.states), "observations")

    def make_contrast_samples(self,):
        posi_ratio = Configs.upperbound
        nega_ratio = Configs.lowerbound
        posi_nums = int(posi_ratio*self.num_states)
        nega_nums = int(nega_ratio*self.num_states)

        self.positive_nums = posi_nums
        self.negative_nums = nega_nums

        states_idxs = list(range(  self.num_states  ))

        # high to low sorted
        sorted_idx, sorted_value = self.sort_by_values( np.array(states_idxs), np.array( self.vlaues_of_states)  )

        posi_idxs = sorted_idx[:posi_nums]
        nega_idxs = sorted_idx[-nega_nums:]

        random.shuffle(posi_idxs)
        random.shuffle(nega_idxs)

        self.positive_state_idxs = posi_idxs
        self.negative_state_idxs = nega_idxs

        return

    def get_positive_samples(self, ):

        positive_idxs_idx = (np.random.randint(self.positive_nums, size = Configs.subbatchsize*Configs.horizon))
        positive_idxs = self.positive_state_idxs[positive_idxs_idx]
        positive_states = self.states_normed[positive_idxs]
        positive_rewards = self.vlaues_of_states[positive_idxs]
        positive_states = positive_states.reshape( Configs.subbatchsize, Configs.horizon, -1 )
        positive_rewards = positive_rewards.reshape( Configs.subbatchsize, Configs.horizon, -1 )

        return positive_states, positive_rewards
    
    def get_negative_samples(self, ):
        if Configs.lowerbound > 0:
            negative_idxs_idx = (np.random.randint(self.negative_nums, size = Configs.subbatchsize*Configs.horizon  ))
            negative_idxs = self.negative_state_idxs[negative_idxs_idx]
            negative_states = self.states_normed[negative_idxs]
            negative_rewards = self.vlaues_of_states[negative_idxs]
            negative_states = negative_states.reshape( Configs.subbatchsize, Configs.horizon, -1 )
            negative_rewards = negative_rewards.reshape( Configs.subbatchsize, Configs.horizon, -1 )
        else:
            negative_states = self.placeholder
            negative_rewards = self.placeholder
        return negative_states, negative_rewards
    
    

class MixValueDataset(MixDataset):
    '''
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:,None]
        self.normed = False
        if normed:
            self.vmin, self.vmax = self._get_bounds()
            self.normed = True

    def _get_bounds(self):
        print('[ datasets/sequence ] Getting value dataset bounds...', end=' ', flush=True)
        vmin = np.inf
        vmax = -np.inf
        for i in range(len(self.indices)):
            value = self.__getitem__(i).values.item()
            vmin = min(value, vmin)
            vmax = max(value, vmax)
        print('✓')
        return vmin, vmax

    def normalize_value(self, value):
        ## [0, 1]
        normed = (value - self.vmin) / (self.vmax - self.vmin)
        ## [-1, 1]
        normed = normed * 2 - 1
        return normed

    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        path_ind, start, end = self.indices[idx]
        rewards = self.fields['rewards'][path_ind, start:]
        discounts = self.discounts[:len(rewards)]
        value = (discounts * rewards).sum()
        if self.normed:
            value = self.normalize_value(value)
        value = np.array([value], dtype=np.float32)
        value_batch = ValueBatch(*batch, value)
        return value_batch


class MixValueDataset_plan1_diffuser(MixValueDataset):
    def __init__(self, *args, discount=0.99, normed=False, **kwargs):
        super().__init__(*args, discount=discount, normed=normed, **kwargs)