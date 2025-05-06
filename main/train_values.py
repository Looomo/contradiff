import diffuser.utils as utils

import os
import sys
import inspect
import torch
import wandb
import sys
import inspect
import random
import numpy as np
import torch

import sys
import inspect
import random
import numpy as np
import torch
from base import Parser
from datetime import datetime
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


def train_value():

    args = Parser()

    args = args.parse_args('values')
    if args.horizon == 4:
        args.dim_mults = (1,4,8)
    elif args.horizon == 32:
        args.dim_mults = (1,2, 4,8)
    print("dim_mults:", args.dim_mults)


    args.n_train_steps = int(args.n_train_steps)
    args.loader = "datasets.MixValueDataset"
    sync_dict(args)
    Configs.set_from_dict(args._dict)
    Configs.add_extra("dataset", args.dataset)
    training_logfile =  os.path.join(Configs.savepath, f"training_log.log")
    Configs.logger = MyLogger(training_logfile)
    
    if Configs.wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project=f"cdiffuser@{args.branch}|{args.task}",
            name=f"{args.dataset}|Seed{args.seed}|H{args.horizon}|ER{args.expert_ratio}|EXD{args.exp_dataset}|{str(datetime.now())}",
            # track hyperparameters and run metadata
            config=Configs._dict
        )


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
        ## value-specific kwargs
        discount=args.discount,
        termination_penalty=args.termination_penalty,
        normed=args.normed,
    )

    render_config = utils.Config(
        args.renderer,
        savepath=(args.savepath, 'render_config.pkl'),
        env=args.dataset,
    )

    dataset = dataset_config()
    renderer = render_config()

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim

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
        device=args.device,
    )

    diffusion_config = utils.Config(
        args.diffusion,
        savepath=(args.savepath, 'diffusion_config.pkl'),
        horizon=args.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=args.n_diffusion_steps,
        loss_type=args.loss_type,
        device=args.device,
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
        label_freq=int(args.n_train_steps // args.n_saves),
        save_parallel=args.save_parallel,
        results_folder=args.savepath,
        bucket=args.bucket,
        n_reference=args.n_reference,
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

    # print('Testing forward...', end=' ', flush=True)
    # batch = utils.batchify(dataset[0], device = args.device)

    # loss, _ = diffusion.loss(*batch)
    # loss.backward()
    # print('✓')

    #-----------------------------------------------------------------------------#
    #--------------------------------- main loop ---------------------------------#
    #-----------------------------------------------------------------------------#

    n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

    # for i in range(n_epochs):
    #     print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_train_steps)


# if __name__ == "__main__":
#     Parser.dataset = "antmaze-large-play-v0"
#     # Parser.dataset = "walker2d-medium-v2"
#     Parser.branch = "diffuser"
#     Parser.seed = 1000
#     Parser.vis_normed = True
#     train_value()

if __name__ == '__main__':
    
    Parser.branch = "plan1_diffuser"
    Parser.dataset = "halfcheetah-medium-replay-v2"
    Parser.exp_dataset = "expert"
    Parser.load_iter = -1
    Parser.nums_eval = 1
    Parser.save_planned = 0
    Parser.seed = 1000
    Parser.expert_ratio = 0.2
    Configs.autoload = False
    Configs.skipbranch = True
    # Configs.autoload =  True
    train_value()