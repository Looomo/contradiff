import socket

# from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

# args_to_watch = [
#     ('myprefix','test/'),
#     ('prefix', ''),
#     ('horizon', 'H'),
#     ('n_diffusion_steps', 'T'),
#     ## value kwargs
#     ('discount', 'd'),
# ]



logbase = 'logs'

#
params_str_plan14b = "NB{args.nblocks}_NH{args.nheads}_DT{args.dmodel}_C{args.with_cond}_HL{args.history_length}_H{args.horizon}_T{args.n_diffusion_steps}_A{args.appendix}_EC{args.dimenc}_DR{args.tdropout}_SEED{args.seed}/\
_U{args.upperbound}_L{args.lowerbound}_E{args.contrastiveembd}_R{args.reduce_method}_W{args.contrastweigth:.1f}_AC{args.act}_CE{args.conembver}_PF{args.posifixratio}_NF{args.negafixratio}_CR{args.contrastratio}"


# plan14bf e 
params_str_plan14bf = "NB{args.nblocks}_NH{args.nheads}_DT{args.dmodel}_C{args.with_cond}_HL{args.history_length}_H{args.horizon}_T{args.n_diffusion_steps}_A{args.appendix}_EC{args.dimenc}_DR{args.tdropout}_SEED{args.seed}/\
_U{args.upperbound}_L{args.lowerbound}_E{args.contrastiveembd}_R{args.reduce_method}_W{args.contrastweigth:.1f}_AC{args.act}_CE{args.conembver}_PF{args.posifixratio}_NF{args.negafixratio}_CR{args.contrastratio}_SLOPE{int(args.slope)}"

# f1 f2 and after
params_str_plan14f1 = "NB{args.nblocks}_NH{args.nheads}_DT{args.dmodel}_C{args.with_cond}_HL{args.history_length}_H{args.horizon}_T{args.n_diffusion_steps}_A{args.appendix}_EC{args.dimenc}_DR{args.tdropout}_SEED{args.seed}/\
_U{args.upperbound}_L{args.lowerbound}_E{args.contrastiveembd}_R{args.reduce_method}_W{args.contrastweigth:.1f}_AC{args.act}_CE{args.conembver}_PF{args.posifixratio}_NF{args.negafixratio}_CR{args.contrastratio}_SLOPE{args.slope}_SUBSEQL{args.subseq_length}"


# 15
params_str_plan15bf = "SEED{args.seed}_EXR{args.expert_ratio}_EXD{args.exp_dataset}/\
_U{args.upperbound}_L{args.lowerbound}_E{args.contrastiveembd}_R{args.reduce_method}_W{args.contrastweigth:.1f}_AC{args.act}_CE{args.conembver}_PF{args.posifixratio}_NF{args.negafixratio}_CR{args.contrastratio}"

params_str_plan15f1 = "SEED{args.seed}_EXR{args.expert_ratio}_EXD{args.exp_dataset}/\
_U{args.upperbound}_L{args.lowerbound}_E{args.contrastiveembd}_R{args.reduce_method}_W{args.contrastweigth:.1f}_AC{args.act}_CE{args.conembver}_PF{args.posifixratio}_NF{args.negafixratio}_CR{args.contrastratio}_SLOPE{args.slope}_SUBSEQL{args.subseq_length}"


params_str_plan15_diffuser = "SEED{args.seed}_EXR{args.expert_ratio}_EXD{args.exp_dataset}"


params_str_plan1_diffuser = "H{args.horizon}_SEED{args.seed}_EXR{args.expert_ratio}_EXD{args.exp_dataset}"

params_str_plan1_hard = "H{args.horizon}_SEED{args.seed}_EXR{args.expert_ratio}_EXD{args.exp_dataset}/\
_U{args.upperbound:.2f}_L{args.lowerbound:.2f}_R{args.reduce_method}_W{args.contrastweigth:.1f}_AC{args.act}_CE{args.conembver}"

params_str_plan2_hard = "H{args.horizon}_SEED{args.seed}_EXR{args.expert_ratio}_EXD{args.exp_dataset}/\
_L{args.lowerbound:.2f}_R{args.reduce_method}_W{args.contrastweigth:.1f}_Sim{args.metrics}"

params_str_plan1_nomix = "H{args.horizon}_SEED{args.seed}/\
_L{args.lowerbound:.2f}_R{args.reduce_method}_W{args.contrastweigth:.1f}_Sim{args.metrics}"

plan3 = "H{args.horizon}_SEED{args.seed}_EXR{args.expert_ratio}_EXD{args.exp_dataset}/\
_U{args.upperbound:.2f}_L{args.lowerbound:.2f}_R{args.reduce_method}_W{args.contrastweigth:.1f}_Sim{args.metrics}_BETA{args.beta}"

params_strs = {
    'plan14e' :params_str_plan14bf,
    'diffuser':"H{args.horizon}_T{args.n_diffusion_steps}",
    'plan14bf':params_str_plan14bf,
    'plan14b':params_str_plan14b,
    'plan14f1':params_str_plan14f1,
    'plan14f2':params_str_plan14f1,
    'plan15bf':params_str_plan15bf,
    'plan15f1':params_str_plan15f1,
    'plan15_diffuser':params_str_plan15_diffuser,
    'plan15fbf':params_str_plan15bf,
    'plan15ff1':params_str_plan15f1,
    'plan15f_diffuser':params_str_plan15_diffuser,
    # ↓ for rebuilt plans
    'plan1_diffuser':params_str_plan1_diffuser,
    'plan1_hard':params_str_plan1_hard,
    'plan2_hard':params_str_plan2_hard,
    'plan1_nomix':params_str_plan1_nomix,
    'plan2_nomix':params_str_plan1_nomix,
    'plan1_hardnce': params_str_plan1_hard,
    'plan2_hardnce':params_str_plan2_hard,
    'plan3':plan3,
}

values_params_str_plan1_diffuser = "SEED{args.seed}_EXR{args.expert_ratio}_EXD{args.exp_dataset}"

value_params_strs={
    'plan1_diffuser':values_params_str_plan1_diffuser,
    # 'plan1_diffuser':params_str_plan1_diffuser,
}



paths = {
    'diffusion/defaults' : 'diffusion/defaults_', # + params_str,
    'values/defaults': 'values/defaults_H{args.horizon}_T{args.n_diffusion_steps}_d{args.discount}',
    'plans/': 'plans/_VB{args.valuebranch}_VS{args.valueseed}_',
}

value_path = ''

addin = {

        'branch': "myprefix",
        'basepath': '/tmp/',  
        'appendix': "Z",


        'train_sep': False,
        'env_type': 'gt',
        'load_ie': True,
        'embd_for_ie': 256,
        'envweight' : 1.,
        'invweight': 1.,
        'inv_weight_in_sep' : 1.,
        'env_weight_in_sep' : 1.,
        

        'lowerbound': 0.2,
        'upperbound': 0.65,
        'temperature': 0.1,
        'subbatchsize': 32,
        'contrastiveembd': 256,
        'contrastweigth': 0.5,
        'reduce_method': "mean",
        'returns_scale': 400,
        'contrast_discount': 0.99,
        'include_returns': False,
        'act': "Identity",
        'tau': 0.5,
        'gamma': 12,
        'slope': 200,

        # transformer embedding
        'dmodel': 512, 
        'nblocks': 6,
        'nheads': 8,
        'tdropout': 0.1, 
        'history_length': 64,
        'with_cond': False,
        'test': False,
        'testplan': '1',


        # embedding diffusion

        'dimenc': 64,
        'recover': False,
        
        'recover': False,
        'nums_eval': 10,

        'batched': False,
        'conembver': 'state',

        'posifixratio': 0.2,
        'negafixratio': 0.2,
        'contrastratio': 0.2,

        'vis_normed': True, 
        'vis': True,

        "save_planned": 0,
        "save_diffusion": 0,
        "subseq_length": 80000,


        "eval_log_appendix": "",
        "eval_log_file": "",
        "seed_idx": 0,
        "evalseed": 0,
        'expert_ratio': 0.01,
        'exp_dataset': None,
        'wandb': False,

        'valuebranch': "plan1_diffuser",
        'valueseed': 1000,
        'save_contrast_splits': False,
        "metrics": "canberra", # euclidean canberra
        'beta': 0.5,
        'testplan': None,
        "tag": None,
}

base = {
    'diffusion': {
        ## model
        'task': 'diffusion',
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 32,
        'n_diffusion_steps': 20,
        'action_weight': 10,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 2, 4, 8),
        'attention': False,
        'renderer': 'utils.MuJoCoRenderer',

        ## dataset
        'loader': 'datasets.SequenceDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'clip_denoised': False,
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': logbase,
        'prefix': 'diffusion/defaults',
        # 'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'l2',
        'n_train_steps': 1e6,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 20000,
        'sample_freq': 20000,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': None,
        'dim': 32


    },

    'values': {
        'task': 'values',
        'model': 'models.ValueFunction',
        'diffusion': 'models.ValueDiffusion',
        'horizon': 32,
        'n_diffusion_steps': 20,
        'dim_mults': (1, 2, 4, 8),
        'renderer': 'utils.MuJoCoRenderer',

        ## value-specific kwargs
        'discount': 0.997,
        'termination_penalty': -100,
        'normed': False,

        ## dataset
        'loader': 'datasets.ValueDataset',
        'normalizer': 'GaussianNormalizer',
        'preprocess_fns': [],
        'use_padding': True,
        'max_path_length': 1000,

        ## serialization
        'logbase': logbase,
        'prefix': 'values/defaults',
        # 'exp_name': watch(args_to_watch),

        ## training
        'n_steps_per_epoch': 10000,
        'loss_type': 'value_l2',
        'n_train_steps': 200e3,
        'batch_size': 32,
        'learning_rate': 2e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 1000,
        'sample_freq': 0,
        'n_saves': 5,
        'save_parallel': False,
        'n_reference': 8,
        'bucket': None,
        'device': 'cuda',
        'seed': None,


        'valuebranch': None,
        'evalseed': None,
        "guide_scale": -1.0,
        'dim': 32
    },

    'plan': {
        'task': 'plan',
        'guide': 'sampling.ValueGuide',
        'policy': 'sampling.GuidedPolicy',
        'max_episode_length': 1000,
        'batch_size': 64,
        'preprocess_fns': [],
        'device': 'cuda',
        'seed': None,

        ## sample_kwargs
        'n_guide_steps': 2,
        'scale': 0.1,
        't_stopgrad': 2,
        'scale_grad_by_std': True,

        ## serialization
        'loadbase': None,
        'valueloadbase': None,
        'logbase': logbase,
        'prefix': 'plans/',
        # 'exp_name': watch(args_to_watch),
        'vis_freq': 100,
        'max_render': 8,

        ## diffusion model
        'horizon': 32,
        'n_diffusion_steps': 20,

        ## value function
        'discount': 0.997,

        ## loading
        'diffusion_loadpath': paths['diffusion/defaults'],  
        'value_loadpath': paths['values/defaults'],

        'diffusion_epoch': 'latest',
        'value_epoch': 'latest',

        'verbose': False,
        'suffix': '0',


        'load_iter': -1,
        'dim': 32,
    },
}


#------------------------ overrides ------------------------#


hopper_medium_expert_v2 = {
    'plan': {
        'scale': 0.0001,
        't_stopgrad': 4,
    },
}


halfcheetah_medium_replay_v2 = halfcheetah_random_v2 = halfcheetah_medium_v2 = halfcheetah_medium_expert_v2 = {
    'diffusion': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
        'attention': True,
    },
    'values': {
        'horizon': 4,
        'dim_mults': (1, 4, 8),
    },
    'plan': {
        'horizon': 4,
        'scale': 0.001,
        't_stopgrad': 4,
    },
}

# CDFNormalizer

kitchen_complete_v0 = kitchen_partial_v0 = kitchen_mixed_v0 = {
    'diffusion': {
        'normalizer': "CDFNormalizer",
        'dim_mults': (1,2, 4, 8),
        'dim': 32
    },
    'values': {
        'normalizer': "CDFNormalizer",
        'dim_mults': (1,2, 4, 8),
        'dim': 32
    },
    'plan': {
        'normalizer': "CDFNormalizer",
        'dim_mults': (1, 2, 4, 8),
        'dim': 32
    },
}

antmaze_large_play_v0 = antmaze_umaze_v0 = antmaze_medium_play_v0 = {
    'diffusion': {
        'normalizer': "CDFNormalizer",
        'dim_mults': (1, 2, 4, 8),
        'dim': 32
    },
    'values': {
        'normalizer': "CDFNormalizer",
        'dim_mults': (1, 2, 4, 8),
        'dim': 32
    },
    'plan': {
        'normalizer': "CDFNormalizer",
        'dim_mults': (1, 2,  4, 8),
        'dim': 32
    },
}

global executed
executed= False
def sync_addin():
    global executed
    if executed:
        print("Already updated. Exiting.")
        assert False
    base['diffusion'].update(addin)
    base['plan'].update(addin)
    base['values'].update(addin)
    print("Finished adding form dicct:addin to dict:base.")
    executed = True

sync_addin()
