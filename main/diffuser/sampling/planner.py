import multiprocessing
from typing import Any
import torch
import torch.nn as nn
from config.locomotion_config import Configs
import os
import einops
import gym
import numpy as np
import pickle
import wandb
from tqdm import tqdm
class Planner():
    def __init__(self, ):
        self.logfile = os.path.join(Configs.savepath, f"evalresults_LOAD{Configs.load_iter}_TP{Configs.testplan}.log")
        # self.env = env
        

    def __call__(self,policy,  diffusion_experiment, value_experiment, logger, env, eval_nums = 1):
        
        cmd = f"echo  `date` >> {self.logfile}"
        os.system(cmd)


        total = 0.

        for i in range(eval_nums):
            observation = env.reset()

            
            obs = policy.normalizer.normalize(observation, 'observations')
            obs = torch.FloatTensor(obs).to(Configs.device)

            obs_r = einops.repeat(
                obs,
                'd -> repeat d', repeat=Configs.batch_size,).unsqueeze(1)


            cond_chain = [torch.zeros_like(obs_r, device = obs_r.device) for i in range(Configs.history_length + 1)]
            cond_chain.append(obs_r)
            cond_chain = cond_chain[ -(Configs.history_length + 1) :]
            history = torch.cat(cond_chain,1)
            ## observations for rendering
            rollout = [observation.copy()]
            total_reward = 0
            for t in range(env.max_episode_steps):
                
                # t = 90
                if t % 10 == 0: print(Configs.savepath, flush=True)

                ## save state for rendering only
                state = env.state_vector().copy()

                ## format current observation for conditioning
                conditions = {0: observation}
                action, trajs, samples = policy(conditions, batch_size=Configs.batch_size, verbose=Configs.verbose, history = history, t = t)

                ## execute action in environment
                next_observation, reward, terminal, _ = env.step(action)

                # obs = policy.normalizer.normalize(next_observation, 'observations')
                obs = next_observation
                obs = torch.FloatTensor(obs).to(Configs.device)
                obs_r = einops.repeat(obs,'d -> repeat d', repeat=Configs.batch_size,).unsqueeze(1)
                cond_chain.append(obs_r)
                cond_chain = cond_chain[ -(Configs.history_length + 1) :]
                history = torch.cat(cond_chain,1)

                ## print reward and score
                total_reward += reward
                score = env.get_normalized_score(total_reward)
                
                print(
                    f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | scale: {Configs.scale}',
                    flush=True,
                )

                ## update rollout observations
                rollout.append(next_observation.copy())

                ## render every `args.vis_freq` steps
                # logger.log(t, trajs, state, rollout)

                if terminal:
                    break

                observation = next_observation

            ## write results to json file at `args.savepath`
            logger.finish(t, score, total_reward, terminal, diffusion_experiment, value_experiment)
            cmd = f"echo {score} >> {self.logfile}"
            os.system(cmd)

            total += score
        cmd = f"echo '------- Mean:{total/eval_nums} ---------' >> {self.logfile}"
        os.system(cmd)    


class Planner_plan1a(Planner):
    def __init__(self):
        super().__init__()

class Planner_plan1b(Planner):
    def __init__(self):
        super().__init__()

class Planner_plan2a(Planner):
    def __init__(self):
        super().__init__()

from datetime import datetime



class ParallelPlanner(Planner):
    def __init__(self):
        super().__init__()
    
    def __call__(self,policy,  diffusion_experiment, value_experiment, logger, env, eval_nums = 1):


        eval_nums = Configs.nums_eval
        
        cmd = f"echo  `date` >> {self.logfile}"
        os.system(cmd)



        env_list = [gym.make(Configs.dataset, seed = datetime.now().microsecond +i) for i in range(eval_nums)]

        for i in range(1):
            # obsnp_normed
            obs_list = [ policy.normalizer.normalize( env.reset(), 'observations')  for env in env_list]
            observation = obs_list
            for i in range(eval_nums):
                observation[i] = torch.FloatTensor(observation[i]).to(Configs.device)
                observation[i] = einops.repeat(
                                        observation[i],
                                        'd -> repeat d', repeat=Configs.batch_size,).unsqueeze(1)


            observation = torch.cat(observation, 0)
            cond_chain = torch.zeros((  Configs.batch_size*eval_nums, Configs.history_length+1, Configs.observation_dim  ), device = observation.device) 
            action_chain = torch.zeros((  Configs.batch_size*eval_nums, Configs.history_length, Configs.action_dim  ), device = observation.device) 
            # cond_chain.append(obs_r)

            cond_chain = torch.cat(  [ cond_chain  ,observation] , 1  )
            cond_chain = cond_chain[ :,  -(Configs.history_length ):, :]
            history = cond_chain

            action_history = action_chain
            ## observations for rendering
            # rollout = [observation[0].copy()]
            dones = [0 for _ in range(eval_nums)]
            episode_rewards = [0 for _ in range(eval_nums)]
            normed_episode_rewards = [0 for _ in range(eval_nums)]
            cond = observation.squeeze(1)
            t = 0
            while sum(dones) <  eval_nums:

                if t % 10 == 0: print(Configs.savepath, flush=True)
                conditions = {0: cond}

                if Configs.history_length == 0:
                    history = torch.zeros((  Configs.batch_size*eval_nums, Configs.history_length, Configs.observation_dim  ), device = observation.device) 
                    action_history = torch.zeros((  Configs.batch_size*eval_nums, Configs.history_length, Configs.action_dim  ), device = observation.device) 


                actions = policy(conditions, batch_size=Configs.batch_size, verbose=Configs.verbose, history_obs = history, history_act = action_history,  t = t)


                obs_list = []

                for i in range(Configs.nums_eval):
                    this_obs, this_reward, this_done, _ = env_list[i].step(actions[i])
                    obs_list.append(this_obs)
                    if this_done:
                        if dones[i] == 1:
                            pass
                        else:
                            dones[i] = 1
                            episode_rewards[i] += this_reward
                            
                            print(f"Episode ({i}): {episode_rewards[i]}  Normed:{normed_episode_rewards[i]}")
                            cmd = f"echo {normed_episode_rewards[i]} >> {self.logfile}"
                            os.system(cmd)

                    else:
                        if dones[i] == 1:
                            pass
                        else:
                            episode_rewards[i] += this_reward

                    normed_episode_rewards[i] = env_list[i].get_normalized_score(episode_rewards[i])

                observation = obs_list
                actions_list = actions
                for i in range(eval_nums):
                    observation[i] = torch.FloatTensor(policy.normalizer.normalize(observation[i], 'observations')).to(Configs.device)
                    actions_list[i] = torch.FloatTensor(policy.normalizer.normalize(actions_list[i], 'actions')).to(action_history.device)
                    actions_list[i] = einops.repeat(
                                            actions_list[i],
                                            'd -> repeat d', repeat=Configs.batch_size,).unsqueeze(1)
                    observation[i] = einops.repeat(
                                            observation[i],
                                            'd -> repeat d', repeat=Configs.batch_size,).unsqueeze(1)

                action_cond = torch.cat(actions_list, 0)
                observation = torch.cat(observation, 0)
                cond = observation.squeeze(1)
                # action_cond = action_cond.squeeze(1)



                cond_chain = torch.cat(  [ cond_chain  ,observation] , 1  )
                cond_chain = cond_chain[ :,  -(Configs.history_length ):, :]
                history = cond_chain

                action_history = torch.cat( [action_history,action_cond ], 1 )
                action_history = action_history[:,-(Configs.history_length ) :, :]

                infos = " ".join( f"{i:4.4f}" for i in normed_episode_rewards)
                print(f"[ {t} / 1000 ]  [{infos}] ")

                t += 1

                if t >= 1000: break

            print(f"average_ep_reward: {np.mean(episode_rewards)}, normed: {np.mean(normed_episode_rewards)}, std_ep_reward: {np.std(episode_rewards)}")
            # print(f'average_ep_reward:{np.mean(episode_rewards)},  normed:{np.mean(normed_episode_rewards)},  std_ep_reward:{np.std(episode_rewards)} ')
        
        cmd = f"echo '------- Mean:{np.mean(normed_episode_rewards)} ---------' >> {self.logfile}"
        os.system(cmd)    


class Planner_plan3a(ParallelPlanner):
    def __init__(self):
        super().__init__()
    





class Planner_plan5a(Planner_plan3a):
    def __init__(self):
        super().__init__()

class Planner_plan6(Planner_plan3a):
    def __init__(self):
        super().__init__()


class Planner_plan6a(Planner_plan6):
    def __init__(self):
        super().__init__()

class Planner_plan6b(Planner_plan6):
    def __init__(self):
        super().__init__()

class Planner_plan6c(Planner_plan6):
    def __init__(self):
        super().__init__()

# 测试为什么加embedding就会变差

class Planner_plan6T2S(Planner_plan6):
    def __init__(self):
        super().__init__()


class Planner_plan6T2N(Planner_plan6):
    def __init__(self):
        super().__init__()

class Planner_plan6T2M(Planner_plan6):
    def __init__(self):
        super().__init__()

class Planner_plan6F2S(Planner_plan6):
    def __init__(self):
        super().__init__()

class Planner_plan6F2N(Planner_plan6):
    def __init__(self):
        super().__init__()

class Planner_plan6F2M(Planner_plan6):
    def __init__(self):
        super().__init__()


class Planner_plan6F2Norm(Planner_plan6):
    def __init__(self):
        super().__init__()


class Planner_plan6T2Norm(Planner_plan6):
    def __init__(self):
        super().__init__()





class Planner_plan6T1S(Planner):
    def __init__(self):
        super().__init__()
class Planner_plan6T1N(Planner_plan6):
    def __init__(self):
        super().__init__()

class Planner_plan6T1M(Planner_plan6):
    def __init__(self):
        super().__init__()

class Planner_plan6F1S(Planner_plan6):
    def __init__(self):
        super().__init__()

class Planner_plan6F1N(Planner_plan6):
    def __init__(self):
        super().__init__()

class Planner_plan6F1M(Planner_plan6):
    def __init__(self):
        super().__init__()


class Planner_plan6F1Norm(Planner_plan6):
    def __init__(self):
        super().__init__()


class Planner_plan6T1Norm(Planner_plan6):
    def __init__(self):
        super().__init__()



class Planner_plan7(Planner_plan6):
    def __init__(self):
        super().__init__()
        # self.normalizer = normalizer

class Planner_plan7aN(Planner_plan7):
    def __init__(self):
        super().__init__()

class Planner_plan7aS(Planner_plan7):
    def __init__(self):
        super().__init__()

class Planner_plan7aM(Planner_plan7):
    def __init__(self):
        super().__init__()

class Planner_plan7aNorm(Planner_plan7):
    def __init__(self):
        super().__init__()




class Planner_plan7dN(Planner_plan7):
    def __init__(self):
        super().__init__()

class Planner_plan7dS(Planner_plan7):
    def __init__(self):
        super().__init__()

class Planner_plan7dM(Planner_plan7):
    def __init__(self):
        super().__init__()

class Planner_plan7dNorm(Planner_plan7):
    def __init__(self):
        super().__init__()



class Planner_plan8a(Planner_plan7):
    def __init__(self):
        super().__init__()

import random
def set_seed(seed):
    if seed is not int:
        Configs.logger.log(f"Skipinh to ste seed {seed}")
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import socket
class Planner_single_thread():
    def __init__(self, token = "", step = None):
        curr_time = str(datetime.now()).replace(' ', '_').replace(':','-')        

        self.label = step if step is not None else Configs.load_iter
        logfile = os.path.join(Configs.savepath,f"eval_log_LOAD{self.label}_{curr_time}_{token}.log")
        self.logfile = os.path.join(Configs.savepath, f"evalresults_LOAD{self.label}_{curr_time}_{token}.log" )
        Configs.eval_logger.logfile = logfile
        Configs.eval_logger.log(f"Task activated at {socket.gethostname()}.")
        Configs.eval_logger.log(f"Date: {curr_time} | Eval Token: {token}")

    def __call__(self, policy, diffusion_exp, value_exp, logger,  env, evalseed = None):

        env.seed(evalseed)
        set_seed(evalseed)
        
        cmd = f"echo  `date` >> {self.logfile}"
        cmd = f"echo  'Evalseed:{evalseed}' >> {self.logfile}"
        os.system(cmd)

        Configs.eval_logger.log(f'↓↓↓ EvalSeed:{evalseed} ↓↓↓',flush=True)

        # total = 0.

        # for i in range(eval_nums):
        traj_file = os.path.join(Configs.savepath, f"trajs_{Configs.dataset}_LOAD{self.label}_ES{Configs.evalseed}_SCALE{Configs.scale}.pkl")
        diffusion_file = os.path.join(Configs.savepath, f"diffusion_{Configs.dataset}_LOAD{self.label}_ES{Configs.evalseed}_SCALE{Configs.scale}.pkl")
        observation = env.reset()
        total_reward = 0
        traj_to_save = []
        diffusion_to_save = []
        infostr = {
            'Reward' :       f"####TensorboardInfos####|Reward|EvalSeed@{evalseed}|Step@{self.label}|",
            'total_Reward' : f"####TensorboardInfos####|total_Reward|EvalSeed@{evalseed}|Step@{self.label}|",
            'normed_score' : f"####TensorboardInfos####|normed_score|EvalSeed@{evalseed}|Step@{self.label}|",
        }
        for t in range(env.max_episode_steps):
            
            conditions = {0: observation}
            action, trajs = policy(conditions, batch_size=Configs.batch_size, verbose=Configs.verbose)
            
            
            next_observation, reward, terminal, _ = env.step(action)

            ## print reward and score
            total_reward += reward
            score = env.get_normalized_score(total_reward)
            if t % 10 == 0:
                Configs.eval_logger.log(
                    f'[ {t}/{env.max_episode_steps} ] | [ Seed = {evalseed} ] | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | scale: {Configs.scale} | {Configs.savepath}',
                    flush=True,
                )
            # Configs.eval_logger.log(f'####TensorboardInfos####|{t}|EvalSeed@{evalseed}|Step@{self.label}|Reward@{reward:.2f}|total_Reward@{total_reward:.2f}|normed_score@{score:.4f}',flush=True)
            infostr['Reward']  = f"{infostr['Reward']}{t}@{reward:.5f}#"
            infostr['total_Reward']  = f"{infostr['total_Reward']}{t}@{total_reward:.5f}#"
            infostr['normed_score']  = f"{infostr['normed_score']}{t}@{score:.5f}#"

            infos = {
                'Reward': reward,
                'total_Reward': total_reward,
                'normed_score': score
            }
            # if Configs.wandb:
            #     if t % 20 == 0:
            #         wandb.log(infos)

            # obs_pred_unnormed = policy.normalizer.unnormalize(trajs[0].observations, 'observations')
            # action_pred_unnormed = policy.normalizer.unnormalize(trajs[0].actions, 'actions')
            if Configs.save_diffusion:
                # print("1111")
                chains = trajs[1]
                diffusion_to_save_cell = {
                    'diffusion':chains,
                    'normed_reward': score,
                    't': t    # current step
                }
                diffusion_to_save.append(diffusion_to_save_cell)

            if Configs.save_planned:
                # print("2222")
                traj_to_save_cell = {
                    'observation_current':observation,       
                    'observation_next_pred':trajs[0].observations[:,1,:],  
                    'observation_next_pred_horizon':trajs[0].observations, 
                    'observation_next_benchmark':next_observation, 

                    'action_pred':trajs[0].actions[:,0,:],  
                    'action_pred_horizon':trajs[0].actions,
                    # 'action_next_benchmark':[],

                    'reward_pred':trajs[0].values,          
                    'r_t':reward,  #  instant return @ step t
                    't': t ,   # current RL step
                    'normed_reward': score, 
                    'decription': "observation_current: un-normed, before execution"\
                    "observation_next_pred: next obs predicted  un-normed" \
                    "observation_next_pred_horizon: reconed obs  un-normed" \
                    "observation_next_benchmark: real next  un-normed" \
                    "r_t:  instant reward" \
                    "normed_reward:  normed score"
                }

                traj_to_save.append(traj_to_save_cell)
            end_step = t
            if terminal:
                break

            observation = next_observation
        Configs.eval_logger.log(
                    f'[ {t}/{env.max_episode_steps} ] | [ Seed = {evalseed} ] | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | scale: {Configs.scale}',
                    flush=True,
                )
        Configs.eval_logger.log(
                    f'[ End ] | [ Seed = {evalseed} ]',
                    flush=True,
                )
        for k in infostr.keys():
            Configs.eval_logger.log(infostr[k][:-1])
        if Configs.save_planned:
            with open(traj_file, "wb") as f:
                pickle.dump(traj_to_save, f)
        if Configs.save_diffusion:
            with open(diffusion_file, "wb") as f:
                pickle.dump(diffusion_to_save, f)

        ## write results to json file at `args.savepath`
        # logger.finish(t, score, total_reward, terminal, diffusion_experiment)
        # cmd = f"echo {score} >> {self.logfile}"
        
        info = f"###SepEval###|1000000|EvalMean:{score:.10f}|EvalVariance:{0.0}"
        Configs.logger.log(f"{info}")
        cmd = f"echo  '####TensorboardInfos####|{score:.10f}' >> {self.logfile}"
        os.system(cmd)
        # total += score
        info = {
            "end_step": end_step,
            "seed": evalseed,
            "score": score
        }
        return info




class Planner_vis():
    def __init__(self, token = "", step = None):
        curr_time = str(datetime.now()).replace(' ', '_').replace(':','-')
        self.label = step if step is not None else Configs.load_iter
        self.logfile = os.path.join(Configs.savepath, f"evalresults_LOAD{self.label}_{curr_time}_{token}.log" )

    def __call__(self, policy, diffusion_exp, value_exp, logger,  env, evalseed = None):

        env.seed(evalseed)
        set_seed(evalseed)
        
        cmd = f"echo  `date` >> {self.logfile}"
        cmd = f"echo  'Evalseed:{evalseed}' >> {self.logfile}"
        os.system(cmd)

        Configs.eval_logger.log(f'↓↓↓ EvalSeed:{evalseed} ↓↓↓',flush=True)

        # total = 0.

        # for i in range(eval_nums):
        traj_file = os.path.join(Configs.savepath, f"trajs_{Configs.dataset}_LOAD{self.label}_ES{Configs.evalseed}_SCALE{Configs.scale}.pkl")
        observation = env.reset()
        total_reward = 0
        traj_to_save = []
        diffusion_to_save = []
        for t in tqdm(range(env.max_episode_steps)):
            
            conditions = {0: observation}
            action, trajs = policy(conditions, batch_size=Configs.batch_size, verbose=Configs.verbose)
            
            
            next_observation, reward, terminal, _ = env.step(action)

            ## print reward and score
            total_reward += reward
            score = env.get_normalized_score(total_reward)
            if Configs.save_diffusion:
                # print("1111")
                chains = trajs[1]
                diffusion_to_save_cell = {
                    'diffusion':chains,
                    'normed_reward': score,
                    't': t    # current step
                }
                diffusion_to_save.append(diffusion_to_save_cell)

            if Configs.save_planned:
                # print("2222")
                traj_to_save_cell = {
                    'observation_current':observation,          
                    'observation_next_pred':trajs[0].observations[:,1,:],   
                    'observation_next_pred_horizon':trajs[0].observations,  
                    'observation_next_benchmark':next_observation,  
                    'action_pred':trajs[0].actions[:,0,:], 
                    'action_pred_horizon':trajs[0].actions,
                    # 'action_next_benchmark':[],

                    'reward_pred':trajs[0].values,          
                    'r_t':reward,  #  instant return @ step t
                    't': t ,   # current RL step
                    'normed_reward': score,   
                    'decription': "observation_current: un-normed, before execution"\
                    "observation_next_pred: next obs predicted  un-normed" \
                    "observation_next_pred_horizon: reconed obs  un-normed" \
                    "observation_next_benchmark: real next  un-normed" \
                    "r_t:  instant reward" \
                    "normed_reward:  normed score"
                }

                traj_to_save.append(traj_to_save_cell)
            end_step = t
            if terminal:
                break
            observation = next_observation
        Configs.eval_logger.log(
                    f'[ {t}/{env.max_episode_steps} ] | [ Seed = {evalseed} ] | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | scale: {Configs.scale}',
                    flush=True,
                )
        Configs.eval_logger.log(
                    f'[ End ] | [ Seed = {evalseed} ]',
                    flush=True,
                )
        if Configs.save_planned:
            Configs.eval_logger.log(
                    f'Saving traj to {traj_file}',
                    flush=True,
                )
            with open(traj_file, "wb") as f:
                pickle.dump(traj_to_save, f)        
            
            Configs.eval_logger.log(
                    f'Saved.',
                    flush=True,
                )
        info = {
            "end_step": end_step,
            "seed": evalseed,
            "score": score
        }
        return info






import copy
class Planner_plan9_batched():
    def __init__(self, ):
        self.logfile = os.path.join(Configs.savepath, f"evalresults_LOAD{Configs.load_iter}_TP{Configs.testplan}_{str(datetime.now()).replace(' ', '_').replace(':','-')}_batched.log")
        # self.env = env
    def __call__(self,policy,  diffusion_experiment, value_experiment, logger, env, eval_nums = 1):


        eval_nums = Configs.nums_eval
        
        cmd = f"echo  `date` >> {self.logfile}"
        os.system(cmd)



        env_list = [gym.make(Configs.dataset, seed = datetime.now().microsecond +i) for i in range(eval_nums)]


        obs_list = [ policy.normalizer.normalize( env.reset(), 'observations')  for env in env_list]
        observation = obs_list
        for i in range(eval_nums):
            observation[i] = torch.FloatTensor(observation[i]).to(Configs.device)
            observation[i] = einops.repeat(
                                    observation[i],
                                    'd -> repeat d', repeat=Configs.batch_size,).unsqueeze(1)


        observation = torch.cat(observation, 0)

        dones = [0 for _ in range(eval_nums)]
        episode_rewards = [0 for _ in range(eval_nums)]
        normed_episode_rewards = [0 for _ in range(eval_nums)]
        cond = observation.squeeze(1)
        t = 0
        while sum(dones) <  eval_nums:
            done_idx = []
            if t % 10 == 0: print(Configs.savepath, flush=True)
            conditions = {0: cond}


            actions = policy(conditions, batch_size=Configs.batch_size, verbose=Configs.verbose)


            obs_list = []

            for i in range(Configs.nums_eval):
                this_obs, this_reward, this_done, _ = env_list[i].step(actions[i])
                obs_list.append(this_obs)
                if this_done:
                    done_idx.append(i)
                    if dones[i] == 1:
                        pass
                    else:
                        dones[i] = 1
                        episode_rewards[i] += this_reward
                        
                        # print(f"Episode ({i}): {episode_rewards[i]}  Normed:{normed_episode_rewards[i]}")
                        cmd = f"echo {normed_episode_rewards[i]} >> {self.logfile}"
                        os.system(cmd)

                else:
                    if dones[i] == 1:
                        pass
                    else:
                        episode_rewards[i] += this_reward

                normed_episode_rewards[i] = env_list[i].get_normalized_score(episode_rewards[i])

            observation = obs_list

            for i in range(eval_nums):
                observation[i] = torch.FloatTensor(policy.normalizer.normalize(observation[i], 'observations')).to(Configs.device)
                observation[i] = einops.repeat(
                                        observation[i],
                                        'd -> repeat d', repeat=Configs.batch_size,).unsqueeze(1)

            observation = torch.cat(observation, 0)
            cond = observation.squeeze(1)
            # action_cond = action_cond.squeeze(1)

            infos = " ".join( f"{i:4.4f}" for i in normed_episode_rewards)
            don = " ".join( f"{i:4.4f}" for i in done_idx)
            print(f"[ {t} / 1000 ]  [{infos}] === env [{don}] terminated.")

            t += 1

            if t >= 1000: break

            # print(f"average_ep_reward: {np.mean(episode_rewards)}, normed: {np.mean(normed_episode_rewards)}, std_ep_reward: {np.std(episode_rewards)}")
        print(f"----> Mean: {np.mean(normed_episode_rewards):4.4f} <----" )
        cmd = f"echo '------- Mean:{np.mean(normed_episode_rewards)} ---------' >> {self.logfile}"
        os.system(cmd)


# base = Planner_plan9_batched if Configs.batched else Planner_plan9_singlt
class Planner_plan9():
    def __init__(self):
        super().__init__()
        self.planner = Planner_plan9_batched() if Configs.batched else Planner_plan9_singlt()
        self.logfile = self.planner.logfile
    def __call__(self,policy,  diffusion_experiment, value_experiment, logger, env, eval_nums = 1):
        return self.planner(policy,  diffusion_experiment, value_experiment, logger, env, eval_nums)

class Planner_plan9a(Planner_plan9):
    def __init__(self):
        super().__init__()

class Planner_plan9b(Planner_plan9):
    def __init__(self):
        super().__init__()

class Planner_plan9c(Planner_plan9):
    def __init__(self):
        super().__init__()

Planner_plan14bf = \
Planner_diffuser = \
Planner_plan14g = \
Planner_plan14f2 = \
Planner_plan14f1 = \
Planner_plan14e = \
Planner_plan14d3 = \
Planner_plan14d1 = \
Planner_plan14d2 = \
Planner_plan14b = \
Planner_plan15a = \
Planner_plan14a2 = \
Planner_plan14a1 = \
Planner_plan12 = \
Planner_plan11 = \
Planner_plan10d = \
Planner_plan10b = \
Planner_plan10a = \
Planner_plan10 = \
Planner_plan9d = \
Planner_plan9afn = \
Planner_plan9af = \
Planner_plan9fn = \
Planner_plan9f = \
Planner_plan9dfn = \
Planner_plan9df = \
Planner_plan9bfn = \
Planner_plan9bf = \
Planner_plan9cfn = \
Planner_plan9cf = \
Planner_plan9
