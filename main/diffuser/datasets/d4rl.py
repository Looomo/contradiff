import os
import collections
import numpy as np
import gym
import pdb

# from contextlib import (
#     contextmanager,
#     redirect_stderr,
#     redirect_stdout,
# )

# @contextmanager
# def suppress_output():
#     """
#         A context manager that redirects stdout and stderr to devnull
#         https://stackoverflow.com/a/52442331
#     """
#     with open(os.devnull, 'w') as fnull:
#         with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
#             yield (err, out)

# with suppress_output():
#     ## d4rl prints out a variety of warnings
import d4rl

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        return name
    # with suppress_output():
    #     wrapped_env = gym.make(name)
    wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env

def get_dataset(env):
    dataset = env.get_dataset()

    # if 'antmaze' in str(env).lower():
    #     ## the antmaze-v0 environments have a variety of bugs
    #     ## involving trajectory segmentation, so manually reset
    #     ## the terminal and timeout fields
    #     dataset = antmaze_fix_timeouts(dataset)
    #     dataset = antmaze_scale_rewards(dataset)
    #     get_max_delta(dataset)

    return dataset

def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset(env)
    dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset

    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1



def sequence_dataset_plain(env, preprocess_fn):

    dataset = get_dataset(env)
    # dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset
    all_data = []
    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            all_data.append(episode_data)
            data_ = collections.defaultdict(list)

        episode_step += 1

    return all_data, dataset

# 返回值： 1. 所有的切分的trajectory  2. 原始的dataset
# 注意： 没有做padding
def sequence_dataset_mix(env):

    dataset = get_dataset(env)
    # dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset
    all_data = []
    episode_step = 0
    start = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            # if done_bool:
            #     print("what the hell")
            end = i
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            episode_data['start'] = start
            episode_data['end'] = end
            episode_data['accumulated_reward'] = np.sum(episode_data['rewards'])
            start = end + 1
            all_data.append(episode_data)
            data_ = collections.defaultdict(list)

        episode_step += 1

    return all_data, dataset


def sequence_dataset_maze2d(env):

    dataset = get_dataset(env)
    # dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    all_data = []
    episode_step = 0
    start = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])
        if done_bool:
            end = i
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            episode_data['start'] = start
            episode_data['end'] = end
            episode_data['accumulated_reward'] = np.sum(episode_data['rewards'])
            start = end + 1
            all_data.append(episode_data)
            data_ = collections.defaultdict(list)

        episode_step += 1

    return all_data, dataset


from tqdm import tqdm

def sequence_dataset_mix_kitchen(env):

    dataset = get_dataset(env)
    # dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    use_timeouts = False
    all_data = []
    episode_step = 0
    start = 0
    for i in tqdm(range(N)):

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])
        
        if dataset['terminals'][i]:
            # if done_bool:
            #     print("what the hell")
            end = i
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            episode_data['start'] = start
            episode_data['end'] = end
            episode_data['accumulated_reward'] = None
            start = end + 1
            all_data.append(episode_data)
            data_ = collections.defaultdict(list)

        episode_step += 1

    return all_data, dataset


#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode
