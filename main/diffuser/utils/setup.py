import os
import importlib
import random
import numpy as np
import torch
from tap import Tap
import pdb
from config.locomotion import paths, params_strs, value_params_strs
from config.locomotion_config import Configs
from .serialization import mkdir
from .git_utils import (
    get_git_rev,
    save_git_diff,
)
def lazy_fstring(template, args):
    ## https://stackoverflow.com/a/53671539
    return eval(f"f'{template}'")    # 可能只是为了修正~
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

# def watch(args_to_watch):
#     def _fn(args):
#         exp_name = []
#         for key, label in args_to_watch:
#             if not hasattr(args, key):
#                 continue
#             val = getattr(args, key)
#             if type(val) == dict:
#                 val = '_'.join(f'{k}-{v}' for k, v in val.items())
#             exp_name.append(f'{label}{val}')
#         exp_name = '_'.join(exp_name)
#         exp_name = exp_name.replace('/_', '/')
#         exp_name = exp_name.replace('(', '').replace(')', '')
#         exp_name = exp_name.replace(', ', '-')
#         return exp_name
#     return _fn

def lazy_fstring(template, args):
    ## https://stackoverflow.com/a/53671539
    return eval(f"f'{template}'")    # 可能只是为了修正~

class Parser(Tap):


    def load_params(self, args):
        params = {
            'halfcheetah-medium-expert-v2':{
                'lowerbound': 0.05,
                'upperbound': 0.65,
                'slope': 1600.0,
                'contrastweigth': 0.1,
                'branch':'plan14f1',
                "seed": 'None'
            },
            'hopper-medium-expert-v2':{
                'lowerbound': 0.35,
                'upperbound': 0.65,
                'slope': 1400.0,
                'contrastweigth': 0.001,
                'branch':'plan14f1',
                'seed': 'None'
            },
            'walker2d-medium-expert-v2':{
                'lowerbound': 0.1,
                'upperbound': 0.65,
                'slope': 200,
                'contrastweigth': 0.001,
                'branch':'plan14bf',
                'seed': 'None'
            },

            'halfcheetah-medium-v2':{
                'lowerbound': 0.2,
                'upperbound': 0.85,
                'slope': 700.0,
                'contrastweigth': 0.01,
                'branch':'plan14f1',
                'seed': 'None'
            },
            'hopper-medium-v2':{
                'lowerbound': 0.2,
                'upperbound': 0.65,
                'slope': 800.0,
                'contrastweigth': 0.001,
                'branch':'plan14f1',
                'seed': 1000
            },
            'walker2d-medium-v2':{
                'lowerbound': 0.2,
                'upperbound': 0.65,
                'slope': 700.0,
                'contrastweigth': 0.1,
                'branch':'plan14f1',
                'seed': 'None'
            },

            'walker2d-random-v2':{
                'lowerbound': 0.2,
                'upperbound': 0.65,
                'slope': 700.0,
                'contrastweigth': 0.1,
                'branch':'plan14f1',
                'seed': 'None'
            },
            'hopper-random-v2':{
                'lowerbound': 0.2,
                'upperbound': 0.65,
                'slope': 800.0,
                'contrastweigth': 0.001,
                'branch':'plan14f1',
                'seed': 1000
            },
            'halfcheetah-random-v2':{
                'lowerbound': 0.2,
                'upperbound': 0.85,
                'slope': 700.0,
                'contrastweigth': 0.01,
                'branch':'plan14f1',
                'seed': 'None'
            },

            'halfcheetah-medium-replay-v2':{
                'lowerbound': 0.4,
                'upperbound': 0.65,
                'slope': 200,
                'contrastweigth': 0.1,
                'branch':'plan14b',
                'act': "Sigmoid"
            },
            'hopper-medium-replay-v2':{
                'lowerbound': 0.2,
                'upperbound': 0.55,
                'slope': 900.0,
                'contrastweigth': 0.001,
                'branch':'plan14f1',
                'seed': None
            },
            'walker2d-medium-replay-v2':{
                'lowerbound': 0.05,
                'upperbound': 0.6,
                'slope': 200,
                'contrastweigth': 0.1,
                'branch':'plan14bf',
                'seed': None
            },

            'maze2d-umaze-v1':{
                'lowerbound': 0.2,
                'upperbound': 5,
                'slope': 1e15,
                'contrastweigth': 0.1,
            },
            'maze2d-medium-v1':{
                'lowerbound': 0.02,
                'upperbound': 0.1,
                'slope': 1e15,
                'contrastweigth': 0.1,
            },
            'maze2d-large-v1':{
                'lowerbound': 0.01,
                'upperbound': 0.6,
                'slope': 1e15,
                'contrastweigth': 0.1,
            },

            'kitchen-mixed-v0':{
                'lowerbound': 0.1,
                'upperbound': 0.25,
                'slope': 1600.0,
                'contrastweigth': 0.1,
                'seed': 100,
                'branch':'plan14bf',
            },

            'kitchen-partial-v0':{
                'lowerbound': 0.1,
                'upperbound': 0.4,
                'slope': 1600.0,
                'contrastweigth': 0.1,
                'seed': 100,
                'branch':'plan14bf',
            },
        }

        to_load = params[args.dataset]
        for k in to_load.keys():
            if Configs.skipbranch:
                if k == "branch":
                    print(f"Skipping load branch ({to_load[k]}). Keeping as {args.branch}.")
                    continue
            setattr(args, k, to_load[k])

        return args

    def save(self):
        fullpath = os.path.join(self.savepath, 'args.json')
        print(f'[ utils/setup ] Saved args to {fullpath}')
        super().save(fullpath, skip_unpicklable=True)

    def parse_args(self, experiment=None, overwrite = None, data = None):
        args = super().parse_args(known_only=False)
        ## if not loading from a config script, skip the result of the setup
        if not hasattr(args, 'config'): return args
        args = self.read_config(args, experiment)

        

        if overwrite is not None:
            
            for k in overwrite.keys():
                print(f"Warning: overwriting args: args.{k}={getattr(args,k)} -> {overwrite[k]}")
                setattr(args,k, overwrite[k])

        self.add_extras(args)

        if data is not None:
            if args.branch in data.keys():
                cell = data[args.branch]
                print(f"Overloading {args.branch}")
                cell = data[args.branch]
                for k in cell.keys():
                    v = cell[k]
                    if hasattr(args, k):
                        print(f"Overloading Parser.{k} = {getattr(args,k)} -> {v}")
                    else:
                        print(f"Warning: Skip loading {k}")
                    setattr(args, k, v)
                if 'load_path' in cell.keys():
                    params_strs[args.branch] = cell['load_path']
        
        if Configs.autoload:
            args = self.load_params(args)
        self.prepare_branch(args)
        # if args.branch == "diffuser" and hasattr(args, 'diffusion_loadpath'):
        #     args.diffusion_loadpath = "diffusion/defaults_H{args.horizon}_T{args.n_diffusion_steps}"

        self.logbase = os.path.join(args.basepath, args.logbase)
        self.eval_fstrings(args)  # 处理类似于 f:{args.seed}之类的字符串

        # args.branch=f"{args.reduce_method}_{ 'mlp' if args.contrastiveembd>0 else 'identity' }"

        self.set_seed(args)
        # self.get_commit(args)  # git相关，可以不用
       
        # self.generate_exp_name(args)
        # self.mkdir(args)
        self.set_save_path(args)
        self.set_loadbase(args)  #设置loadbase为logbase
        # self.save_diff(args)
        return args
    def prepare_branch(self, args):
        paths['diffusion/defaults'] += params_strs[args.branch]
        paths['values/defaults'] += value_params_strs[args.valuebranch]
        paths['plans/'] += (params_strs[args.branch])
        if args.task == 'plan':
            args.diffusion_loadpath = paths['diffusion/defaults']
            args.value_loadpath = paths['values/defaults']
    def read_config(self, args, experiment):
        '''
            Load parameters from config file
        '''
        dataset = args.dataset.replace('-', '_')
        print(f'[ utils/setup ] Reading config: {args.config}:{dataset}')
        module = importlib.import_module(args.config)
        params = getattr(module, 'base')[experiment]

        if hasattr(module, dataset) and experiment in getattr(module, dataset):
            print(f'[ utils/setup ] Using overrides | config: {args.config} | dataset: {dataset}')
            overrides = getattr(module, dataset)[experiment]
            params.update(overrides)
        else:
            print(f'[ utils/setup ] Not using overrides | config: {args.config} | dataset: {dataset}')

        self._dict = {}
        for key, val in params.items():
            if not hasattr(args, key):
                setattr(args, key, val)
            self._dict[key] = getattr(args, key)

        return args

    def add_extras(self, args):
        '''
            Override config parameters with command-line arguments
        '''
        extras = args.extra_args
        if not len(extras):
            return

        print(f'[ utils/setup ] Found extras: {extras}')
        assert len(extras) % 2 == 0, f'Found odd number ({len(extras)}) of extras: {extras}'
        for i in range(0, len(extras), 2):
            key = extras[i].replace('--', '')
            val = extras[i+1]
            assert hasattr(args, key), f'[ utils/setup ] {key} not found in config: {args.config}'
            old_val = getattr(args, key)
            old_type = type(old_val)
            print(f'[ utils/setup ] Overriding config | {key} : {old_val} --> {val}')
            if val == 'None':
                val = None
            elif val == 'latest':
                val = 'latest'
            elif old_type in [bool, type(None)]:
                try:
                    val = eval(val)
                except:
                    print(f'[ utils/setup ] Warning: could not parse {val} (old: {old_val}, {old_type}), using str')
            else:
                val = old_type(val)
            setattr(args, key, val)
            self._dict[key] = val

    def eval_fstrings(self, args):
        for key, old in self._dict.items():
            if type(old) is str and old[:2] == 'f:':
                val = old.replace('{', '{args.').replace('f:', '')
                new = lazy_fstring(val, args)
                print(f'[ utils/setup ] Lazy fstring | {key} : {old} --> {new}')
                setattr(self, key, new)
                self._dict[key] = new

    def set_seed(self, args):
        if not hasattr(args, 'seed') or args.seed is None or type(args.seed) is not int:
            return
        print(f'[ utils/setup ] Setting seed: {args.seed}')
        set_seed(args.seed)

    def set_loadbase(self, args):
        if hasattr(args, 'loadbase') and args.loadbase is None:
            print(f'[ utils/setup ] Setting loadbase: {args.logbase}')
            args.loadbase = args.logbase

    def set_save_path(self,args):
        # if 'diffusion' in args.prefix or 'plans' in args.prefix: 
        #     exp_name = lazy_fstring(   paths['diffusion']  )
        # if 'values' in args.prefix: 
        #     exp_name = lazy_fstring(   paths['values']  )

        exp_name = lazy_fstring(   paths[args.prefix] , args )
        # if args.branch == "diffuser" and hasattr(args, 'diffusion_loadpath'):
        #     exp_name = lazy_fstring( "plans/_H{args.horizon}_T{args.n_diffusion_steps}_EVALSEED{args.evalseed}", args )

        args.savepath = os.path.join(args.logbase, args.branch, args.dataset, exp_name)
        self._dict['savepath'] = getattr(args, 'savepath')
        if mkdir(args.savepath):
            print(f'[ utils/setup ] Made savepath: {args.savepath}')
        # self._dict['savepath'] = args.savepath

    # def generate_exp_name(self, args):
    #     if not 'exp_name' in dir(args):
    #         return
    #     exp_name = getattr(args, 'exp_name')
    #     if callable(exp_name):
    #         exp_name_string = exp_name(args)
    #         print(f'[ utils/setup ] Setting exp_name to: {exp_name_string}')
    #         setattr(args, 'exp_name', exp_name_string)
    #         self._dict['exp_name'] = exp_name_string

    # def mkdir(self, args):
    #     if 'logbase' in dir(args) and 'dataset' in dir(args) and 'exp_name' in dir(args):
    #         args.savepath = os.path.join(args.logbase, args.dataset, args.exp_name)
    #         self._dict['savepath'] = args.savepath
    #         if 'suffix' in dir(args):
    #             args.savepath = os.path.join(args.savepath, args.suffix)
    #         if mkdir(args.savepath):
    #             print(f'[ utils/setup ] Made savepath: {args.savepath}')
    #         self.save()

    # def get_commit(self, args):
    #     args.commit = get_git_rev()

    # def save_diff(self, args):
    #     try:
    #         save_git_diff(os.path.join(args.savepath, 'diff.txt'))
    #     except:
    #         print('[ utils/setup ] WARNING: did not save git diff')
