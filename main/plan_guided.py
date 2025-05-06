import multiprocessing
import pdb
import sys
import os
current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
grandparent_directory = os.path.dirname(parent_directory)
sys.path.append(grandparent_directory)
import wandb
from datetime import datetime
import socket
import diffuser.sampling as sampling
from diffuser.sampling import *
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
from base import Parser
from diffuser.utils import lazy_fstring
import einops
from myutils import MyLogger, convert_eval, convert_sepeval, convert_evalresults


if __name__ == '__main__':
    current_file_name = inspect.getfile(inspect.currentframe())
    path = os.path.dirname(os.path.abspath(current_file_name))
    os.chdir(path)
    print(f"Changing working dir to {path}.")

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

def sync_dict(args):

    for k in args._dict.keys():
        val = getattr(args, k)
        args._dict[k] = val
    
import matplotlib
matplotlib.use('Agg')
import shutil
def parseargs(config = None):

    

    args = Parser()

    args = args.parse_args('plan')

    if args.horizon == 4:
        args.dim_mults = (1,4,8)
    elif args.horizon == 32:
        args.dim_mults = (1,2, 4,8)

    args.diffusion = f"models.CDiffusion_{args.branch}"
    # args.model = f"models.TransCondTemporalUnet_{args.branch}"
    args.loader = f"datasets.MixDataset_{args.branch}"    


    system_infos = {
        'user_name' : os.environ['USER'] if 'USER' in os.environ.keys() else "Unable to acquire $USER",
        'host' : socket.gethostname(),
        'user_name' : os.environ['USER'] if 'USER' in os.environ.keys() else "Unable to acquire $USER",
        'jobid':os.environ['SLURM_JOB_ID'] if 'SLURM_JOB_ID' in os.environ else "Unable to acquire 'SLURM_JOB_ID'.",
        'jobname':os.environ['SLURM_JOB_NAME'] if 'SLURM_JOB_NAME' in os.environ else "Unable to acquire 'SLURM_JOB_NAME'.",
        'devices' : os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else "Unable to acquire 'CUDA_VISIBLE_DEVICES'.",
        
    }
    system_infos['slurm_job_cfg_file'] = f"/home/{system_infos['user_name']}/slurm/{system_infos['jobname']}.slurm",



    if not 'VALBASE' in os.environ:
        args.valueloadbase = f"/home/{system_infos['user_name']}/sync_recv/valuebase"
    else:
        args.valueloadbase = os.environ['VALBASE']
    # args.valuebranch = 'plan1_diffuser'
    # template = args.value_loadpath.replace("args.seed", "args.valueseed")
    # args.valueloadbase = os.path.join(args.logbase,args.valuebranch,args.dataset,  eval(f"f'{template}'"))
    
    
    #-----------------------------debug only--------------------------------------#

  

    #-----------------------------------------------------------------------------#
    
    # args.with_cond = True
    args.test = True

    if args.guide_scale >= 0:
        args.scale = args.guide_scale
    args.expert_ratio = float(args.expert_ratio)
    sync_dict(args)
    Configs.set_from_dict(args._dict)
    Configs.dataset = args.dataset

    eval_logfile =  os.path.join(Configs.savepath, f"eval_log.log")

    Configs.eval_logger = MyLogger(eval_logfile)
    Configs.logger = MyLogger(os.path.join(Configs.savepath, f"eval_log_final.log"))

    

    if Configs.wandb:
        tags = [system_infos['user_name']]
        if Configs.tag is not None:
            tags.append(Configs.tag)
        Configs.run = wandb.init(
            # set the wandb project where this run will be logged
            project=f"cdiffuser@{args.branch}|{args.task}",
            # project=f"test",
            name=f"{args.dataset}|Seed{args.seed}|H{args.horizon}|ER{args.expert_ratio}|UL{args.upperbound}#{args.lowerbound}|{str(datetime.now())}",
            tags=tags,
            # track hyperparameters and run metadata
            config=Configs._dict
        )

    

    Configs.logger.log(f"======== Basic Infos ========")
    for k in system_infos.keys():
        Configs.logger.log(f"{k}: {system_infos[k]}")
    Configs.logger.log(f"=============================")

    return args
    #-----------------------------------------------------------------------------#
    #---------------------------------- loading ----------------------------------#
    #-----------------------------------------------------------------------------#
    # args.seed = args.evalseed
    ## load diffusion model and value function from disk
def evaluate(args):
    old_version_path = os.path.join(args.loadbase, args.branch, args.dataset, lazy_fstring( args.diffusion_loadpath.replace(":.2f", ":.1f").replace("_R{args.reduce_method}", "_E{args.contrastiveembd}_R{args.reduce_method}")  , args))
    old_version_path2 = os.path.join(args.loadbase, args.branch, args.dataset, lazy_fstring( args.diffusion_loadpath.replace(":.2f", ":.0f").replace("_R{args.reduce_method}", "_E{args.contrastiveembd}_R{args.reduce_method}")  , args))
    if os.path.exists(old_version_path):
        Configs.logger.log(f"Found old version of diffusion path. Renemeing to new version.")
        new_version = os.path.join(args.loadbase, args.branch, args.dataset, lazy_fstring( args.diffusion_loadpath, args))
        shutil.move(old_version_path, new_version)
        Configs.logger.log(f"{old_version_path} --> {new_version}")
    if os.path.exists(old_version_path2):
        Configs.logger.log(f"Found old version of diffusion path. Renemeing to new version.")
        new_version = os.path.join(args.loadbase, args.branch, args.dataset, lazy_fstring( args.diffusion_loadpath, args))
        shutil.move(old_version_path2, new_version)
        Configs.logger.log(f"{old_version_path2} --> {new_version}")

    
    diffusion_experiment = utils.load_diffusion(   # 通过加载训练时保存的model dataset等配置文件重新加载模型。依赖diffusion_loadpath与~
        args.loadbase, args.branch, args.dataset, lazy_fstring( args.diffusion_loadpath  , args),
        epoch=args.load_iter, seed=args.evalseed, device= args.device,
        args = args
    )
    if "nomix" in args.branch: 
        args.value_loadpath = args.value_loadpath.split("SEED")[0]
        args.valuebranch = "diffuser"
    else:
        args.value_loadpath =  args.value_loadpath.replace("args.seed", "args.valueseed")
    value_experiment = utils.load_diffusion(
        args.valueloadbase, args.valuebranch, args.dataset, lazy_fstring(args.value_loadpath, args),
        epoch=-1, seed=args.evalseed,device= args.device,
        args = args
    )

    ## ensure that the diffusion model and value function are compatible with each other
    utils.check_compatibility(diffusion_experiment, value_experiment)

    diffusion = diffusion_experiment.ema
    dataset = diffusion_experiment.dataset
    renderer = diffusion_experiment.renderer

    ## initialize value guide
    value_function = value_experiment.ema
    guide_config = utils.Config(args.guide, model=value_function, verbose=False)
    guide = guide_config()

    logger_config = utils.Config(
        utils.Logger,
        renderer=renderer,
        logpath=args.savepath,
        vis_freq=args.vis_freq,
        max_render=args.max_render,
    )

    ## policies are wrappers around an unconditional diffusion model and a value guide
    policy_config = utils.Config(
        args.policy,
        guide=guide,
        scale=args.scale,
        diffusion_model=diffusion,
        normalizer=dataset.normalizer,
        preprocess_fns=args.preprocess_fns,
        ## sampling kwargs
        sample_fn=sampling.n_step_guided_p_sample,
        n_guide_steps=args.n_guide_steps,
        t_stopgrad=args.t_stopgrad,
        scale_grad_by_std=args.scale_grad_by_std,
        verbose=False,
    )

    logger = logger_config()
    policy = policy_config()


    #-----------------------------------------------------------------------------#
    #--------------------------------- main loop ---------------------------------#
    #-----------------------------------------------------------------------------#

    # logfile = os.path.join(Configs.savepath, f"evalresults_LOAD{args.load_iter}_TP{args.testplan}.log")

    
    # planner = eval(f"Planner_{Configs.branch}()")
    planner = Planner_single_thread()

    Configs.eval_logger.log( f"Loaded epoch: {diffusion_experiment.trainer.step} with rpoch :{diffusion_experiment.epoch}."  )
    names = ['seed', 'score', 'end_step']
    scores = []
    for seed in range(args.nums_eval):
        info = planner(policy, diffusion_experiment, value_experiment, logger, dataset.env, evalseed = seed)
        scores.append([info['seed'], info['score'], info['end_step']])

        wandb.log( {"score": info['score']}  )
    
    scores.append([-1, np.mean(scores,0)[1], np.std(scores,0)[1]])
    if Configs.wandb:
        Configs.run.log(   {   "Summary": wandb.Table( data=scores, columns=names )    }  )
    convert_evalresults(Configs.savepath)

    return



from base import load_best

if __name__ == '__main__':
    
    Parser.branch = "plan1_nomix"
    Parser.dataset = "walker2d-medium-v2"
    Parser.exp_dataset = "expert"
    Parser.load_iter = -1
    Parser.nums_eval = 50
    Parser.save_planned = 0
    Parser.seed = 1000
    Configs.autoload = False
    Configs.skipbranch = True
    Parser.expert_ratio = 0
    Parser.upperbound = 0.1
    Parser.lowerbound = 0.1
    Parser.device = "cuda:3"
    # Configs.autoload =  True
    args = parseargs()
    
    score = evaluate(args)
