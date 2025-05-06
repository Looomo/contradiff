from collections import namedtuple
import torch
import einops
import pdb

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn


Trajectories = namedtuple('Trajectories', 'actions observations values')


class GuidedPolicy:

    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs

    def __call__(self, conditions, batch_size=1, verbose=True):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process
        samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, return_chain = True, **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)
        chain = utils.to_np(samples.chains)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions_normed = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions_normed, 'actions')

        ## extract first action
        action = actions[0, 0]

        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        trajectories = Trajectories(actions, observations, samples.values)
        return action, (trajectories, chain)

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device=self.device)
        conditions = utils.apply_dict(
            einops.repeat,
            conditions,
            'd -> repeat d', repeat=batch_size,
        )
        return conditions

class TransCondGuidedPolicy_main(GuidedPolicy):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)
    
    def __call__(self, conditions, batch_size=1, verbose=True, history = None, t = 0):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)
        assert history is not None
        ## run reverse diffusion process
        samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, history = history, step = t, **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)

        ## extract action [ batch_size x horizon x transition_dim ]
        actions = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')

        ## extract first action
        action = actions[0, 0]

        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        trajectories = Trajectories(actions, observations, samples.values)
        return action, trajectories, samples



class TransCondGuidedPolicy_plan1a(TransCondGuidedPolicy_main):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

    

class TransCondGuidedPolicy_plan2a(TransCondGuidedPolicy_main):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)


# class TransCondGuidedPolicy_plan3a(TransCondGuidedPolicy_main):
#     def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
#         super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

    
#     def __call__(self, conditions, batch_size=1, verbose=True, history_obs = None, history_act = None, t = 0):
#         conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
#         conditions = self._format_conditions(conditions, batch_size)
#         assert history_obs is not None
#         ## run reverse diffusion process
#         self.diffusion_model.eval()
#         samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, history_obs = history_obs, history_act = history_act, step = t, **self.sample_kwargs)
#         trajectories = utils.to_np(samples.trajectories)

#         ## extract action [ batch_size x horizon x transition_dim ]
#         actions = trajectories[:, :, :self.action_dim]
#         actions = self.normalizer.unnormalize(actions, 'actions')

#         ## extract first action
#         action = actions[0, 0]

#         normed_observations = trajectories[:, :, self.action_dim:]
#         observations = self.normalizer.unnormalize(normed_observations, 'observations')

#         trajectories = Trajectories(actions, observations, samples.values)
#         return action, trajectories, samples



class ParallelTransCondGuidedPolicy(TransCondGuidedPolicy_main):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

    # def extract_action(batch):

    def __call__(self, conditions, batch_size=1, verbose=True, history_obs = None, history_act = None, t = 0):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = self._format_conditions(conditions, batch_size)
        assert history_obs is not None
        ## run reverse diffusion process
        self.diffusion_model.eval()
        samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, history_obs = history_obs, history_act = history_act, step = t, **self.sample_kwargs)
        trajectories = samples.trajectories

        ## extract action [ batch_size x horizon x transition_dim ]
        # actions = trajectories[:, :, :self.action_dim]


        actions = [  self.normalizer.unnormalize(traj.cpu().numpy()[:, :, :self.action_dim][0,0]  , 'actions')     for traj in trajectories    ]
        # actions = 

        ## extract first action
        # action = actions[0, 0]

        # normed_observations = trajectories[:, :, self.action_dim:]
        # observations = self.normalizer.unnormalize(normed_observations, 'observations')
        # trajectories = Trajectories(actions, observations, samples.values)
        
        return actions

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        # conditions = utils.apply_dict(
        #     self.normalizer.normalize,
        #     conditions,
        #     'observations',
        # )
        # conditions = utils.to_torch(conditions, dtype=torch.float32, device=self.device)
        # conditions = utils.apply_dict(
        #     einops.repeat,
        #     conditions,
        #     'd -> repeat d', repeat=batch_size,
        # )
        return conditions
    

class TransCondGuidedPolicy_plan3a(ParallelTransCondGuidedPolicy):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)


class TransCondGuidedPolicy_plan5a(TransCondGuidedPolicy_plan3a):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)


class TransCondGuidedPolicy_plan6(TransCondGuidedPolicy_plan5a):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan6a(TransCondGuidedPolicy_plan6):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan6b(TransCondGuidedPolicy_plan6):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan6c(TransCondGuidedPolicy_plan6):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)


# 测试为什么加embedding就会变差

class TransCondGuidedPolicy_plan6T2S(TransCondGuidedPolicy_plan6):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan6T2N(TransCondGuidedPolicy_plan6):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan6T2M(TransCondGuidedPolicy_plan6):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan6F2S(TransCondGuidedPolicy_plan6):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan6F2N(TransCondGuidedPolicy_plan6):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan6F2M(TransCondGuidedPolicy_plan6):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan6F2Norm(TransCondGuidedPolicy_plan6):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan6T2Norm(TransCondGuidedPolicy_plan6):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)


class TransCondGuidedPolicy_plan6T1S(TransCondGuidedPolicy_plan6):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan6T1N(TransCondGuidedPolicy_plan6):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan6T1M(TransCondGuidedPolicy_plan6):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan6F1S(TransCondGuidedPolicy_plan6):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan6F1N(TransCondGuidedPolicy_plan6):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan6F1M(TransCondGuidedPolicy_plan6):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan6F1Norm(TransCondGuidedPolicy_plan6):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan6T1Norm(TransCondGuidedPolicy_plan6):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan7(TransCondGuidedPolicy_plan6):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)


class TransCondGuidedPolicy_plan7aN(TransCondGuidedPolicy_plan7):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan7aS(TransCondGuidedPolicy_plan7):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)
    
class TransCondGuidedPolicy_plan7aM(TransCondGuidedPolicy_plan7):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan7aNorm(TransCondGuidedPolicy_plan7):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)




class TransCondGuidedPolicy_plan7dN(TransCondGuidedPolicy_plan7):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan7dS(TransCondGuidedPolicy_plan7):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)
    
class TransCondGuidedPolicy_plan7dM(TransCondGuidedPolicy_plan7):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

class TransCondGuidedPolicy_plan7dNorm(TransCondGuidedPolicy_plan7):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)


TransCondGuidedPolicy_plan9_singlt = GuidedPolicy
class TransCondGuidedPolicy_plan9_batched:
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        self.guide = guide
        self.diffusion_model = diffusion_model
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs
    def __call__(self, conditions, batch_size=1, verbose=True):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        self.diffusion_model.eval()
        samples = self.diffusion_model(conditions, guide=self.guide, verbose=verbose, **self.sample_kwargs)
        trajectories = samples.trajectories
        actions = [  self.normalizer.unnormalize(traj.cpu().numpy()[:, :, :self.action_dim][0,0]  , 'actions')     for traj in trajectories    ]
        return actions

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

from config.locomotion_config import Configs
# base = TransCondGuidedPolicy_plan9_batched if Configs.batched else TransCondGuidedPolicy_plan9_singlt




class TransCondGuidedPolicy_plan9():
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__()
        self.policy = TransCondGuidedPolicy_plan9_batched(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs) if Configs.batched else TransCondGuidedPolicy_plan9_singlt(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)
        self.guide = self.policy.guide
        self.diffusion_model = self.policy.diffusion_model
        self.normalizer = self.policy.normalizer
        self.action_dim = self.policy.diffusion_model.action_dim
        self.preprocess_fn = self.policy.preprocess_fn
        self.sample_kwargs = self.policy.sample_kwargs
    def __call__(self, conditions, batch_size=1, verbose=True):
        return self.policy(conditions, batch_size, verbose)
    
class TransCondGuidedPolicy_plan9a(TransCondGuidedPolicy_plan9):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)


class TransCondGuidedPolicy_plan9b(TransCondGuidedPolicy_plan9):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)


class TransCondGuidedPolicy_plan9c(TransCondGuidedPolicy_plan9):
    def __init__(self, guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs):
        super().__init__(guide, diffusion_model, normalizer, preprocess_fns, **sample_kwargs)

TransCondGuidedPolicy_plan14bf = \
TransCondGuidedPolicy_diffuser = \
TransCondGuidedPolicy_plan14g = \
TransCondGuidedPolicy_plan14f2 = \
TransCondGuidedPolicy_plan14f1 = \
TransCondGuidedPolicy_plan14e = \
TransCondGuidedPolicy_plan14d3 = \
TransCondGuidedPolicy_plan14b = \
TransCondGuidedPolicy_plan14d2 = \
TransCondGuidedPolicy_plan14d1 = \
TransCondGuidedPolicy_plan15a = \
TransCondGuidedPolicy_plan14a2 = \
TransCondGuidedPolicy_plan14a1 = \
TransCondGuidedPolicy_plan12 = \
TransCondGuidedPolicy_plan11 = \
TransCondGuidedPolicy_plan10d = \
TransCondGuidedPolicy_plan10b = \
TransCondGuidedPolicy_plan10a = \
TransCondGuidedPolicy_plan10 = \
TransCondGuidedPolicy_plan9d = \
TransCondGuidedPolicy_plan9afn = \
TransCondGuidedPolicy_plan9af = \
TransCondGuidedPolicy_plan9dfn = \
TransCondGuidedPolicy_plan9df = \
TransCondGuidedPolicy_plan9bfn = \
TransCondGuidedPolicy_plan9bf = \
TransCondGuidedPolicy_plan9fn = \
TransCondGuidedPolicy_plan9f = \
TransCondGuidedPolicy_plan9cfn = \
TransCondGuidedPolicy_plan9cf = \
TransCondGuidedPolicy_plan9