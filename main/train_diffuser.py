import sys
import os
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
os.chdir(parent_directory)
grandparent_directory = os.path.dirname(parent_directory)
sys.path.append(grandparent_directory)
import socket
import diffuser.utils as utils
import os
import sys
import inspect
import torch
from datetime import datetime
import sys
import inspect
import random
import numpy as np
import torch
from base import Parser
import wandb
import os
import inspect
current_file_name = inspect.getfile(inspect.currentframe())
path = os.path.dirname(os.path.abspath(current_file_name))
os.chdir(path)
print(f"Changing working dir to {path}.")

import matplotlib
matplotlib.use('Agg')

def sync_dict(args):

    for k in args._dict.keys():
        val = getattr(args, k)
        args._dict[k] = val


from config.locomotion_config import Configs
from myutils import MyLogger, convert_log
import copy
def train_diffusion():

    args = Parser()
    

    args = args.parse_args('diffusion')
    if args.horizon == 4:
        args.dim_mults = (1,4,8)
    elif args.horizon == 32:
        args.dim_mults = (1,2, 4,8)

    args.n_train_steps = int(args.n_train_steps)

    
    args.diffusion = f"models.CDiffusion_{args.branch}"
    # args.model = f"models.TransCondTemporalUnet_{args.branch}"
    args.loader = f"datasets.MixDataset_{args.branch}"    

    args.expert_ratio = float(args.expert_ratio)
    sync_dict(args)
    Configs.set_from_dict(args._dict)
    Configs.add_extra("dataset", args.dataset)
    training_logfile =  os.path.join(Configs.savepath, f"training_log.log")
    Configs.logger = MyLogger(training_logfile)

    system_infos = {
        'user_name' : os.environ['USER'] if 'USER' in os.environ.keys() else "Unable to acquire $USER",
        'host' : socket.gethostname(),
        'user_name' : os.environ['USER'] if 'USER' in os.environ.keys() else "Unable to acquire $USER",
        'jobid':os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ else "Unable to acquire 'SLURM_JOB_ID'.",
        'jobname':os.environ['SLURM_JOB_NAME'] if 'SLURM_JOB_NAME' in os.environ else "Unable to acquire 'SLURM_JOB_NAME'.",
        'devices' : os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else "Unable to acquire 'CUDA_VISIBLE_DEVICES'.",
        
    }
    system_infos['slurm_job_cfg_file'] = f"/home/{system_infos['user_name']}/slurm/{system_infos['jobname']}.slurm",
    
    if Configs.wandb:
        tags = [system_infos['user_name']]
        if Configs.tag is not None:
            tags.append(Configs.tag)
        wandb.init(
            # set the wandb project where this run will be logged
            project=f"cdiffuser@{args.branch}|{args.task}",
            name=f"{args.dataset}|Seed{args.seed}|H{args.horizon}|ER{args.expert_ratio}|UL{args.upperbound}#{args.lowerbound}|{str(datetime.now())}",
            tags=tags,
            # track hyperparameters and run metadata
            config=Configs._dict
        )

    

    Configs.logger.log(f"======== Basic Infos ========")
    for k in system_infos.keys():
        Configs.logger.log(f"{k}: {system_infos[k]}")
    Configs.logger.log(f"=============================")

    if Configs.upperbound > Configs.expert_ratio:
        Configs.logger.log( f"!!! Warning: Configs.upperbound > Configs.expert_ratio ({Configs.upperbound:.2f} > {Configs.expert_ratio:.2f}), not a good sign!"  )

    # args.device = 'cuda:1'
    #-----------------------------------------------------------------------------#
    #---------------------------------- dataset ----------------------------------#
    #-----------------------------------------------------------------------------#

    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, 'dataset_config.pkl'),
        env=args.dataset,
        horizon=args.horizon,
        normalizer=args.normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
    )

    render_config = utils.Config(
        args.renderer,
        savepath=(args.savepath, 'render_config.pkl'),
        env=args.dataset
    )

    dataset = dataset_config()
    renderer = render_config()

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim


    #   build config obj
    
    Configs.add_extra("observation_dim", observation_dim)
    Configs.add_extra("action_dim", action_dim)
    Configs.add_extra("transition_dim", observation_dim + action_dim)
    Configs.add_extra("env", copy.deepcopy(dataset.env))
    Configs.savecfg()

    
    #-----------------------------------------------------------------------------#
    #------------------------------ model & trainer ------------------------------#
    #-----------------------------------------------------------------------------#


    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, 'model_config.pkl'),
        horizon=args.horizon,
        transition_dim=observation_dim + action_dim,
        cond_dim=observation_dim,
        dim_mults=args.dim_mults,
        attention=args.attention,
        device=args.device,
        dim = args.dim
    )

    diffusion_config = utils.Config(
        args.diffusion,
        savepath=(args.savepath, 'diffusion_config.pkl'),
        horizon=args.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=args.n_diffusion_steps,
        loss_type=args.loss_type,
        clip_denoised=args.clip_denoised,
        predict_epsilon=args.predict_epsilon,
        ## loss weighting
        action_weight=args.action_weight,
        loss_weights=args.loss_weights,
        loss_discount=args.loss_discount,
        device=args.device
    )

    trainer_config = utils.Config(
        utils.Trainer,
        savepath=(args.savepath, 'trainer_config.pkl'),
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        sample_freq=args.sample_freq,
        save_freq=args.save_freq,
        label_freq=int(200000),
        save_parallel=args.save_parallel,
        results_folder=args.savepath,
        bucket=args.bucket,
        n_reference=args.n_reference,
        log_freq = args.log_freq
    )

    #-----------------------------------------------------------------------------#
    #-------------------------------- instantiate --------------------------------#
    #-----------------------------------------------------------------------------#

    model = model_config()

    diffusion = diffusion_config(model)

    trainer = trainer_config(diffusion, dataset, renderer)


    #-----------------------------------------------------------------------------#
    #------------------------ test forward & backward pass -----------------------#
    #-----------------------------------------------------------------------------#

    utils.report_parameters(model)


    #-----------------------------------------------------------------------------#
    #--------------------------------- main loop ---------------------------------#
    #-----------------------------------------------------------------------------#

    # trainer.train(1000)
    trainer.train(args.n_train_steps)





if __name__ == '__main__':
    # 这里的参数会被命令行参数重载
    Parser.save_planned = 0
    Parser.dataset = "walker2d-medium-v2"
    Parser.exp_dataset = "None"
    Parser.branch = "plan1_nomix"
    Parser.upperbound = 0.1
    Parser.lowerbound = 0
    Parser.seed = 1000
    Parser.vis_normed = True
    Parser.save_diffusion = 0
    Parser.save_planned = 0
    Parser.wandb = False
    # Parser.device = "cuda:1"
    Parser.expert_ratio = 0.0
    Parser.wandb = False
    # Parser.log_freq = 10
    Configs.autoload = False
    Configs.skipbranch = True # 不要从最优参数中load branch. 仅当autoload = True时有效
    
    train_diffusion()