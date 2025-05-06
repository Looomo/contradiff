import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils
import sys
import inspect
import random
import numpy as np
import torch
import os
import sys
import inspect
import torch
from config.locomotion_config import Configs
import copy

current_file_name = inspect.getfile(inspect.currentframe())
path = os.path.dirname(os.path.abspath(current_file_name))

from config.locomotion import base



class Parser(utils.Parser):
        dataset: str = 'halfcheetah-medium-replay-v2'
        config: str = 'config.locomotion'
        device: str = 'cuda:0'
        seed = None
        basepath:str = path
        branch: str = "plan6T2M"
        
        # n_train_steps: int = 1e6

        nblocks: int  = 1
        history_length: int = 20
        
        appendix: str = "Z"
        with_cond: bool = True

        load_iter: int = -1

        test: bool = False
        testplan: str = "1"

        # scale: float = 0.1

        dmodel: int = 512
        
        nheads: int = 8
        dimenc: int = 64

        recover: bool = False
        tdropout: float = 0.1

        batch_size: int = 64
        nums_eval: int = 10

        lowerbound: float = 0.0
        upperbound: float = 0.0


        contrastweigth: float = 0.1
        act: str = "Softmax"
        tau: float = 0.5
        gamma: float = 12
        slope: float = 200
        
        batched: bool = False
        conembver: str = "state" # state/traj
        contrastratio: float = 1.0
        posifixratio: float = 1.0
        negafixratio: float = 1.0
        vis_normed = False
        vis= False

        evalseed: int = None


        save_planned: int = 0
        save_diffusion: int = 0
        subseq_length: int = 60000
        eval_log_appendix: str = ""
        eval_log_file: str = None
        seed_idx: int = 0
        guide_scale: float = -1.0
        logbase:str = "logs"
        dim: int = 32
        exp_dataset: str = None
        expert_ratio: float = 0.01
        wandb: bool = False
        log_freq: int = 1000
        valuebranch: str = "plan1_diffuser"
        valueseed: int = 1000
        save_contrast_splits: bool = False
        metrics: str = 'canberra'
        horizon: int = 32
        dim_mults: tuple = (1,2,4,8)
        beta: float = 0.5
        tag: str = None
        # recover = True

def load_best():
        return