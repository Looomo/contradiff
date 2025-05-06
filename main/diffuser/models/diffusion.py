from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb

import torch.nn.functional as F
# from mingpt import GPT
import einops
import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

from config.locomotion_config import Configs
Sample = namedtuple('Sample', 'trajectories values chains')

from .contrastive_loss import *
from .sample_plans import *

@torch.no_grad()
def default_sample_fn(model, x, cond, t):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


import copy
# class GTEnv():
#     def __init__(self):
#         super().__init__()
#         self.env = copy.deepcopy(Configs.env)

#     def set_env_state_via_obs(self, obs):

#         obs_np = obs

#         state = self.env.wrapped_env.sim.get_state()
#         dim = state.qpos.shape[0]
#         state.qpos[1:] = obs_np[:dim-1]
#         state.qvel[:] = obs_np[dim-1:]
#         self.env.wrapped_env.sim.set_state(state)
#         self.env.wrapped_env.sim.forward()

#         return self.env

#     def forward(self, sa):

#         a = sa[:, :, : Configs.action_dim ]
#         s = sa[:, :, Configs.action_dim : ]
#         obs_next = []

#         batch, horizon, _ = s.shape

#         for b in range(batch):
#             obs_list = []
#             for t in range(horizon):
#                 obs_old = s[b,t,:]
#                 action = a[b,t,:]
#                 self.env = self.set_env_state_via_obs(obs_old.detach().cpu().numpy())
#                 obs, reward, _, _ = self.env.step(action.detach().cpu().numpy())
#                 obs_list.append(obs)
            
#             obs_next.append(obs_list)
        

#         return torch.FloatTensor(obs_next).to(Configs.device)



# class TransionInverseModel(nn.Module):
#     def __init__(self, emb_dim, observation_dim, action_dim ):
#         super().__init__()

#         self.emb_dim = emb_dim
#         self.observation_dim = observation_dim
#         self.action_dim = action_dim


#         if Configs.env_type == 'mlp':
#             self.transion_model = nn.Sequential(
#                 nn.Linear(self.action_dim + self.observation_dim, emb_dim),
#                 nn.ReLU(),
#                 nn.Linear(emb_dim, emb_dim),
#                 nn.ReLU(),
#                 nn.Linear(emb_dim, self.observation_dim),
#             )
#         elif Configs.env_type == 'gpt':
#             gpt_cfg = GPT.get_default_config()
#             gpt_cfg.n_embd = self.action_dim + self.observation_dim
#             gpt_cfg.block_size = Configs.horizon - 1
#             gpt_cfg.device = Configs.device
#             self.transion_model = GPT(gpt_cfg)

#         elif Configs.env_type == 'gt':
#              self.transion_model = GTEnv()

#         self.inv_model = nn.Sequential(
#                 nn.Linear(2 * self.observation_dim, emb_dim),
#                 nn.ReLU(),
#                 nn.Linear(emb_dim, emb_dim),
#                 nn.ReLU(),
#                 nn.Linear(emb_dim, self.action_dim),
#             )
        


#     def loss(self,  x_start, cond, reward):
#         batch, traj, action_and_observ = x_start.shape

#         actions = x_start[:, :, :self.action_dim]
#         stats = x_start[:, :, self.action_dim: ]


#         state_next = stats[:, 1:, :] # 求出每个时刻的下个时刻的状态，第一个和最后一个状态舍弃

#         # 用于训练inverse的数据
#         for_inverse = torch.cat( [ stats[:,:-1,:],  state_next ] , dim=-1)
#         inv_target = actions[:,:-1,:]
        
#         # 用于训练trans的数据
#         for_trans = x_start[:,:-1,:]
#         trans_target = stats[:,1:,:] if Configs.env_type == 'mlp' else x_start[:,1:,:]


#         pred_action = self.inv_model(for_inverse)
#         loss_inv = F.mse_loss(pred_action,inv_target)

#         pred_next_s = self.transion_model(for_trans)
#         loss_trans = F.mse_loss(pred_next_s,trans_target)

#         return 0.5*( Configs.inv_weight_in_sep*loss_inv + Configs.env_weight_in_sep*loss_trans )


#     # def forward(self, input, type_inverse = False):
#     #     if type_inverse:
#     #         assert input.shape[-1] == self.observation_dim*2
#     #         return self.inv_model(input)
#     #     else:
#     #         assert input.shape[-1] == self.action_dim + self.observation_dim
#     #         return self.transion_model(input) if Config.env_type == 'mlp' else self.transion_model.forward(input)[:, :, self.action_dim: ]



class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None
        sample_fn = n_step_guided_p_sample_diffuser
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()

        x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = Configs.batch_size
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, **sample_kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)  # ?

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)
        info['TotalLoss'] = loss
        info['DiffLoss'] = loss
        return loss, info

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)
    
class TransCondGaussianDiffusion_diffuser(GaussianDiffusion):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)


class ValueDiffusion(GaussianDiffusion):

    def p_losses(self, x_start, cond, target, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        pred = self.model(x_noisy, cond, t)

        loss, info = self.loss_fn(pred, target)
        return loss, info

    def forward(self, x, cond, t):
        return self.model(x, cond, t)



def build_mask(step = -1):

    mask_1 = torch.tril(torch.ones(Configs.history_length + Configs.horizon, Configs.horizon + Configs.history_length, device= torch.device(Configs.device)),diagonal=0)#.to(Configs.device)

    if step < 0:
        mask_2 = torch.triu(torch.ones(Configs.history_length + Configs.horizon, Configs.horizon + Configs.history_length, device= torch.device(Configs.device)),diagonal=-Configs.history_length)#.to(Configs.device)
    else:
        mask_2 = torch.triu(torch.ones(Configs.history_length + Configs.horizon, Configs.history_length + Configs.horizon, device= torch.device(Configs.device)),diagonal=-min(Configs.history_length, step))#.to(Configs.device)


    mask = mask_1 * mask_2

    # return None
    return mask



# from ..encoder import get_encoder
class TransCondGaussianDiffusion_main(GaussianDiffusion):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

        self.encoder = torch.nn.Identity()

    
    def p_losses(self, x_start, cond, history, t):

        noise = torch.randn_like(x_start)

        history_embd = self.encoder(history, mask = None)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t, history_embd)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, history = None, step = 0, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device
        # from queue import Queue

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        # history = Queue(maxsize=Configs.history_length)
        # history.put(cond[0])

        
        chain = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, t, history = history, step = step,  **sample_kwargs)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()

        x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)
    

    def p_mean_variance(self, x, cond, t, step, history):
        
        if step < Configs.history_length:
            # generate mask
            mask = torch.zeros(Configs.history_length, Configs.history_length, device = x.device)
            mask[-(step+1):, -(step+1):] = 1
        else:
            mask = torch.ones(Configs.history_length, Configs.history_length, device = x.device)

        history_embd = self.encoder(history, mask = mask)


        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t, history_embd))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    



class TransCondGaussianDiffusion_plan1a(GaussianDiffusion):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

        # self.encoder = get_encoder()
        # self.masks = None
        # self.build_masks()
        # assert self.masks is not None


    def build_masks(self,):
        mask_1 = torch.tril(torch.ones(Configs.history_length + Configs.horizon, Configs.horizon + Configs.history_length, device= torch.device(Configs.device)),diagonal=0)#.to(Configs.device)
        mask_2_last = torch.triu(torch.ones(Configs.history_length + Configs.horizon, Configs.horizon + Configs.history_length, device= torch.device(Configs.device)),diagonal=-Configs.history_length)#.to(Configs.device)
        mask_2s = [
            torch.triu(torch.ones(Configs.history_length + Configs.horizon, Configs.history_length + Configs.horizon, device= torch.device(Configs.device)),diagonal=-step ).unsqueeze(0)
            for step in range(Configs.history_length)
        ]
        mask_2s.append( mask_2_last.unsqueeze(0))
        mask_2s = torch.cat(mask_2s, 0)
        mask_last = mask_1*mask_2_last
        masks = mask_2s*mask_1.unsqueeze(0)
        self.masks = masks
        # self.masks = torch.cat([ masks, mask_last.unsqueeze(0) ], 0)
        # del masks
        # del mask_2s
        # del mask_last

    def build_mask(self, step = -1):
        return self.masks[min(step, Configs.history_length)]

    
    def p_losses(self, x_start, cond, history, t):

        noise = torch.randn_like(x_start)


        history_embd = self.encoder(history, mask = self.build_mask())

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t, history_embd)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, history = None, step = 0, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device
        # from queue import Queue

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        # history = Queue(maxsize=Configs.history_length)
        # history.put(cond[0])

        
        chain = [x]

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            history_ = torch.cat(  [history, x[ :, 1:, self.action_dim:  ]]  , dim = 1  )
            x, values = sample_fn(self, x, cond, t, history = history_, step = step,  **sample_kwargs)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()

        x, values = sort_by_values(x, values)
        chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)
    

    def p_mean_variance(self, x, cond, t, step, history):
        # 这里的history已经把s0加进去了，所以是21
        # if step < Configs.history_length:
        #     # generate mask
        #     mask = torch.zeros(Configs.history_length, Configs.history_length, device = x.device)
        #     mask[-(step+1):, -(step+1):] = 1
        # else:
        #     mask = build_mask()

        history_embd = self.encoder(history, mask = self.build_mask(step))


        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t, history_embd))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    

class TransCondGaussianDiffusion_plan1b(TransCondGaussianDiffusion_plan1a):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)


class TransCondGaussianDiffusion_plan2a(TransCondGaussianDiffusion_plan1a):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)
       
        self.contrastive_embd_layer = nn.Sequential(
                nn.Linear(self.horizon * self.transition_dim, Configs.contrastiveembd),
                nn.Softmax(),
            ) if Configs.contrastiveembd > 0 else nn.Identity()

        self.contrastive_loss_fn = soft_info_nce
    
    def p_losses(self, x_start, cond, positive, negative, history, t):
        noise = torch.randn_like(x_start)

        history_embd = self.encoder(history, mask = self.build_mask() )
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t, history_embd)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)


        if Configs.reduce_method == "mean":
            reduc_weight_posi = torch.ones( x_start.shape[0], Configs.subbatchsize ).to(Configs.device) / Configs.subbatchsize
            reduc_weight_nega = torch.ones( x_start.shape[0], Configs.subbatchsize ).to(Configs.device) / Configs.subbatchsize
        if Configs.reduce_method == "cos_no_norm":
            x_recon_s0 = x_recon[:,0,self.action_dim:]
            posi_s0 = positive[:,:,0,self.action_dim:]
            nega_s0 = negative[:,:,0,self.action_dim:]

            cos_positive = F.cosine_similarity(   x_recon_s0.unsqueeze(1), posi_s0, eps=1e-12     )
            cos_negative = F.cosine_similarity(   x_recon_s0.unsqueeze(1), nega_s0, eps=1e-12     )
            
            reduc_weight_posi = cos_positive + 1
            reduc_weight_nega = cos_negative + 1 

        if Configs.reduce_method == "cos":
            x_recon_s0 = x_recon[:,0,self.action_dim:]
            posi_s0 = positive[:,:,0,self.action_dim:]
            nega_s0 = negative[:,:,0,self.action_dim:]


            cos_positive = F.cosine_similarity(   x_recon_s0.unsqueeze(1), posi_s0, -1,  eps=1e-12     )
            cos_negative = F.cosine_similarity(   x_recon_s0.unsqueeze(1), nega_s0, -1,  eps=1e-12     )

            def linearnorm(input):
                max_val, _ = torch.max(input, -1)
                min_val, _ = torch.min(input, -1)

                normed = (input - min_val.unsqueeze(1)) / (max_val.unsqueeze(1) - min_val.unsqueeze(1))

                normed/= torch.sum( normed, -1 ).unsqueeze(-1)

                return normed


            reduc_weight_posi = linearnorm(cos_positive)
            reduc_weight_nega = linearnorm(cos_negative)

        x_recon_embd  = self.contrastive_embd_layer(einops.rearrange(x_recon, 'b h e -> b (h e)'))
        positive_embd = self.contrastive_embd_layer(einops.rearrange(positive, 'b bs h e -> b bs (h e)'))#.mean(1))
        negative_embd = self.contrastive_embd_layer(einops.rearrange(negative, 'b bs h e -> b bs (h e)'))



        cons_loss = self.contrastive_loss_fn(x_recon_embd, positive_embd, negative_embd,  reduc_weight_posi = reduc_weight_posi, reduc_weight_nega = reduc_weight_nega   )

        info_pack =  {'diff_loss': loss, 'cons_loss': cons_loss}
        info_pack.update(info)
        return  loss+cons_loss*Configs.contrastweigth, info_pack

    # def loss(self, x, cond, positive, negative, history):
    #     batch_size = len(x)
    #     t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
    #     return self.p_losses(x, cond, positive, negative, history, t)



class TransCondGaussianDiffusion_plan3a(TransCondGaussianDiffusion_plan1a):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)


    def p_losses(self, x_start, cond, history_obs, history_act, t):

        noise = torch.randn_like(x_start)


        history_embd = self.encoder(history_obs, history_act, mask = self.build_mask())

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t, history_embd)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info
    

    def p_mean_variance(self, x, cond, t, step, history_obs, history_act):

        history_embd = self.encoder(history_obs, history_act, mask = self.build_mask(step))


        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t, history_embd))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    


    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, history_obs = None, history_act = None, step = 0, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device
        # from queue import Queue

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        # history = Queue(maxsize=Configs.history_length)
        # history.put(cond[0])
        sample_fn = eval(f"n_step_guided_p_sample_{Configs.branch}")
        
        chain = [x]
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            history_obs_ = torch.cat(  [history_obs, x[ :, 1:, self.action_dim:  ]]  , dim = 1  )
            history_act_ = torch.cat(  [history_act, x[ :, :, :self.action_dim  ]]  , dim = 1  )

            x, values = sample_fn(self, x, cond, t, history_obs = history_obs_, history_act = history_act_, step = step,  **sample_kwargs)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()

        x, values = sort_by_values(x, values)
        chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)
    


class TransCondGaussianDiffusion_plan5a(TransCondGaussianDiffusion_plan3a):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)



class ParallelTransCondGaussianDiffusion(TransCondGaussianDiffusion_plan3a):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)
        

    def p_losses(self, x_start, cond, history_obs, history_act, t):

        # stats = x_start[:,:, self.action_dim:]
        # stat_emb = self.state_pre_encoder(stats)
        # cond_0 = self.state_pre_encoder(cond[0])
        noise = torch.randn_like(x_start)


        # history_embd = self.encoder(history_obs, history_act, mask = self.build_mask())

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t, bert = self.encoder, mask = self.build_mask(),  history_obs= history_obs, history_act= history_act)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info
    

    def p_mean_variance(self, x, cond, t, step, history_obs, history_act):

        # history_embd = self.encoder(history_obs, history_act, mask = self.build_mask(step))
        
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t, bert = self.encoder, mask = self.build_mask(step),  history_obs= history_obs, history_act= history_act))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    


    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, history_obs = None, history_act = None, step = 0, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device
        # from queue import Queue

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        # history = Queue(maxsize=Configs.history_length)
        # history.put(cond[0])
        sample_fn = eval(f"n_step_guided_p_sample_{Configs.branch}")
        
        chain = [x]
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            history_obs_ = torch.cat(  [history_obs, x[ :, :, self.action_dim:  ]]  , dim = 1  )
            # history_obs_ = self.state_pre_encoder(history_obs_)
            history_act_ = torch.cat(  [history_act, x[ :, :, :self.action_dim  ]]  , dim = 1  )

            x, values = sample_fn(self, x, cond, t, history_obs = history_obs_, history_act = history_act_, step = step,  **sample_kwargs)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()
        # nums_eval = 
        x_ = []
        vals_ = []
        for i in range(Configs.nums_eval):
            x_s, values_s = sort_by_values(x[i*Configs.batch_size:(i+1)*Configs.batch_size, : ,: ], values[i*Configs.batch_size:(i+1)*Configs.batch_size ])
            x_.append(x_s)
            vals_.append(values_s)
        chain = torch.stack(chain, dim=1)
        return Sample(x_, vals_, chain)


class TransCondGaussianDiffusion_plan6(TransCondGaussianDiffusion_plan3a):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)
        self.encoder = get_encoder()
        self.masks = None
        self.build_masks()
        assert self.masks is not None

class TransCondGaussianDiffusion_plan6a(TransCondGaussianDiffusion_plan6):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)


class TransCondGaussianDiffusion_plan6b(TransCondGaussianDiffusion_plan6):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan6c(TransCondGaussianDiffusion_plan6):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)


# 测试为什么加embedding就会变差
class TransCondGaussianDiffusion_plan6T2S(TransCondGaussianDiffusion_plan6):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan6T2N(TransCondGaussianDiffusion_plan6):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan6T2M(TransCondGaussianDiffusion_plan6):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan6F2S(TransCondGaussianDiffusion_plan6):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan6F2N(TransCondGaussianDiffusion_plan6):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan6F2M(TransCondGaussianDiffusion_plan6):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)


class TransCondGaussianDiffusion_plan6F2Norm(TransCondGaussianDiffusion_plan6):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan6T2Norm(TransCondGaussianDiffusion_plan6):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)



class TransCondGaussianDiffusion_plan6T1S(TransCondGaussianDiffusion_plan6):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan6T1N(TransCondGaussianDiffusion_plan6):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan6T1M(TransCondGaussianDiffusion_plan6):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan6F1S(TransCondGaussianDiffusion_plan6):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan6F1N(TransCondGaussianDiffusion_plan6):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan6F1M(TransCondGaussianDiffusion_plan6):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)


class TransCondGaussianDiffusion_plan6F1Norm(TransCondGaussianDiffusion_plan6):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan6T1Norm(TransCondGaussianDiffusion_plan6):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)



class TransCondGaussianDiffusion_plan7(TransCondGaussianDiffusion_plan6):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan7aN(TransCondGaussianDiffusion_plan7):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan7aS(TransCondGaussianDiffusion_plan7):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan7aM(TransCondGaussianDiffusion_plan7):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan7aNorm(TransCondGaussianDiffusion_plan7):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)




class TransCondGaussianDiffusion_plan7dN(TransCondGaussianDiffusion_plan7):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan7dS(TransCondGaussianDiffusion_plan7):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan7dM(TransCondGaussianDiffusion_plan7):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan7dNorm(TransCondGaussianDiffusion_plan7):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)




class TransCondGaussianDiffusion_plan8a(TransCondGaussianDiffusion_plan7):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan8b(TransCondGaussianDiffusion_plan7):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)


class TransCondGaussianDiffusion_plan9(TransCondGaussianDiffusion_plan3a):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

    
        self.contrastive_embd_layer = nn.Sequential(
                nn.Linear(self.observation_dim, Configs.contrastiveembd),
                eval(f"nn.{Configs.act}()"),
            ) if Configs.contrastiveembd > 0 else nn.Identity()

        self.contrastive_loss_fn = soft_info_nce
    
    def p_losses(self, x_start, cond, positives, negatives, positives_vals, negatives_vals,  history,  t  ):

        # stats = x_start[:,:, self.action_dim:]
        # stat_emb = self.state_pre_encoder(stats)
        # cond_0 = self.state_pre_encoder(cond[0])
        noise = torch.randn_like(x_start)


        # history_embd = self.encoder(history_obs, history_act, mask = self.build_mask())

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)


        # if Configs.reduce_method == "mean":
        reduc_weight_posi = torch.ones( x_start.shape[0], Configs.subbatchsize ).to(Configs.device) / Configs.subbatchsize
        reduc_weight_nega = torch.ones( x_start.shape[0], Configs.subbatchsize ).to(Configs.device) / Configs.subbatchsize


        x_recon_embd  = self.contrastive_embd_layer( x_recon[:, 0, self.action_dim:] )
        positive_embd = self.contrastive_embd_layer( positives )#.mean(1))
        negative_embd = self.contrastive_embd_layer( negatives )



        cons_loss = self.contrastive_loss_fn(x_recon_embd, positive_embd, negative_embd,  reduc_weight_posi = reduc_weight_posi, reduc_weight_nega = reduc_weight_nega   )

        info_pack =  {'diff_loss': loss, 'cons_loss': cons_loss}
        info_pack.update(info)

        loss_all = cons_loss*Configs.contrastweigth + loss

        return loss_all, info_pack
    
    def p_mean_variance(self, x, cond, t):



        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    


    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device
        # from queue import Queue

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        # history = Queue(maxsize=Configs.history_length)
        # history.put(cond[0])
        sample_fn = n_step_guided_p_sample_diffuser
        
        chain = [x]
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            # history_obs_ = torch.cat(  [history_obs, x[ :, 1:, self.action_dim:  ]]  , dim = 1  )
            # history_act_ = torch.cat(  [history_act, x[ :, :, :self.action_dim  ]]  , dim = 1  )

            x, values = sample_fn(self, x, cond, t,  **sample_kwargs)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()

        # for single-thread
        if not Configs.batched:
            x, values = sort_by_values(x, values)
            chain = torch.stack(chain, dim=1)
            return Sample(x, values, chain)
        
        if Configs.batched:
            x_ = []
            vals_ = []
            for i in range(Configs.nums_eval):
                x_s, values_s = sort_by_values(x[i*Configs.batch_size:(i+1)*Configs.batch_size, : ,: ], values[i*Configs.batch_size:(i+1)*Configs.batch_size ])
                x_.append(x_s)
                vals_.append(values_s)
            chain = torch.stack(chain, dim=1)
            return Sample(x_, vals_, chain)


class TransCondGaussianDiffusion_plan9f(TransCondGaussianDiffusion_plan9):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

    def p_losses(self, x_start, cond, positives, negatives, positives_vals, negatives_vals,  history,  t  ):

        # stats = x_start[:,:, self.action_dim:]
        # stat_emb = self.state_pre_encoder(stats)
        # cond_0 = self.state_pre_encoder(cond[0])
        noise = torch.randn_like(x_start)


        # history_embd = self.encoder(history_obs, history_act, mask = self.build_mask())

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)


        # if Configs.reduce_method == "mean":
        reduc_weight_posi = torch.ones( x_start.shape[0], Configs.subbatchsize ).to(Configs.device) / Configs.subbatchsize
        reduc_weight_nega = torch.ones( x_start.shape[0], Configs.subbatchsize ).to(Configs.device) / Configs.subbatchsize


        x_recon_embd  = self.contrastive_embd_layer( x_recon[:, 1, self.action_dim:] )
        positive_embd = self.contrastive_embd_layer( positives )#.mean(1))
        negative_embd = self.contrastive_embd_layer( negatives )



        cons_loss = self.contrastive_loss_fn(x_recon_embd, positive_embd, negative_embd,  reduc_weight_posi = reduc_weight_posi, reduc_weight_nega = reduc_weight_nega   )

        info_pack =  {'diff_loss': loss, 'cons_loss': cons_loss}
        info_pack.update(info)

        loss_all = cons_loss*Configs.contrastweigth + loss

        return loss_all, info_pack

class TransCondGaussianDiffusion_plan9a(TransCondGaussianDiffusion_plan9):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

        self.inputdim =  self.horizon*self.observation_dim  if Configs.conembver=="traj" else   self.observation_dim
        self.contrastive_embd_layer = nn.Sequential(
                nn.Linear( self.inputdim  , Configs.contrastiveembd),
                eval(f"nn.{Configs.act}()"),
            ) if Configs.contrastiveembd > 0 else nn.Identity()

        self.contrastive_loss_fn = InfoNCE(temperature =  Configs.temperature, negative_mode = "paired")  if Configs.conembver=="traj" else  soft_info_nce_traj 

        self.traj_reduce_weight = self.build_traj_reduce_weight()

    def build_traj_reduce_weight(self):
        re_weight = torch.FloatTensor([1 for i in range(Configs.horizon)]).to(Configs.device).reshape(1,1,Configs.horizon)
        traj_reduce_weight = 1/re_weight
        return  traj_reduce_weight #/torch.sum(traj_reduce_weight, -1)
    def p_losses(self, x_start, cond, positives, negatives, positives_vals, negatives_vals,  history,  t  ):

        noise = torch.randn_like(x_start)


        # history_embd = self.encoder(history_obs, history_act, mask = self.build_mask())

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)


        # if Configs.reduce_method == "mean":
        reduc_weight_posi = torch.ones( x_start.shape[0], Configs.subbatchsize ).to(Configs.device) / Configs.subbatchsize
        reduc_weight_nega = torch.ones( x_start.shape[0], Configs.subbatchsize ).to(Configs.device) / Configs.subbatchsize

        # always state level
        if Configs.conembver == "traj":  
            x_recon_embd = self.contrastive_embd_layer(einops.rearrange(x_recon[:, :, self.action_dim:], 'b h e -> b (h e)'))
            positive_embd = self.contrastive_embd_layer(einops.rearrange(positives, 'b bs h e -> b bs (h e)').mean(1))
            negative_embd = self.contrastive_embd_layer(einops.rearrange(negatives, 'b bs h e -> b bs (h e)'))
            cons_loss = self.contrastive_loss_fn(x_recon_embd, positive_embd, negative_embd)
        else:
            x_recon_embd  = self.contrastive_embd_layer( x_recon[:, :, self.action_dim:] )
            positive_embd = self.contrastive_embd_layer( positives )#.mean(1))
            negative_embd = self.contrastive_embd_layer( negatives ) if Configs.lowerbound > 0 else None
            cons_loss = self.contrastive_loss_fn(x_recon_embd, positive_embd, negative_embd,  reduc_weight_posi = reduc_weight_posi, reduc_weight_nega = reduc_weight_nega , traj_reduce_weight = self.traj_reduce_weight  )



        # cons_loss = self.contrastive_loss_fn(x_recon_embd, positive_embd, negative_embd,  reduc_weight_posi = reduc_weight_posi, reduc_weight_nega = reduc_weight_nega , traj_reduce_weight = self.traj_reduce_weight  )


        loss_all = cons_loss*Configs.contrastweigth + loss

        info['DiffLoss'] = loss
        info['TotalLoss'] = loss_all
        info['cons_loss'] = cons_loss

        return loss_all, info


class TransCondGaussianDiffusion_plan9af(TransCondGaussianDiffusion_plan9a):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

    def build_traj_reduce_weight(self):
        re_weight = torch.FloatTensor([1 for i in range(Configs.horizon - 1)]).to(Configs.device).reshape(1,1,Configs.horizon - 1)
        traj_reduce_weight = 1/re_weight
        return  traj_reduce_weight #/torch.sum(traj_reduce_weight, -1)
    
    def p_losses(self, x_start, cond, positives, negatives, positives_vals, negatives_vals,  history,  t  ):

        noise = torch.randn_like(x_start)


        # history_embd = self.encoder(history_obs, history_act, mask = self.build_mask())

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)


        # if Configs.reduce_method == "mean":
        reduc_weight_posi = torch.ones( x_start.shape[0], Configs.subbatchsize ).to(Configs.device) / Configs.subbatchsize
        reduc_weight_nega = torch.ones( x_start.shape[0], Configs.subbatchsize ).to(Configs.device) / Configs.subbatchsize

        x_recon_embd  = self.contrastive_embd_layer( x_recon[:, 1:, self.action_dim:] )
        positive_embd = self.contrastive_embd_layer( positives[:, :, 1:, :] )#.mean(1))
        negative_embd = self.contrastive_embd_layer( negatives[:, :, 1:, :] )



        cons_loss = self.contrastive_loss_fn(x_recon_embd, positive_embd, negative_embd,  reduc_weight_posi = reduc_weight_posi, reduc_weight_nega = reduc_weight_nega , traj_reduce_weight = self.traj_reduce_weight  )

        info_pack =  {'diff_loss': loss, 'cons_loss': cons_loss}
        info_pack.update(info)

        loss_all = cons_loss*Configs.contrastweigth + loss

        return loss_all, info_pack

class TransCondGaussianDiffusion_plan9afn(TransCondGaussianDiffusion_plan9af):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)
    def build_traj_reduce_weight(self):
        re_weight = torch.FloatTensor([1 for i in range(Configs.horizon - 1)]).to(Configs.device).reshape(1,1,Configs.horizon - 1)
        traj_reduce_weight = 1/re_weight
        return  traj_reduce_weight/torch.sum(traj_reduce_weight, -1)

class TransCondGaussianDiffusion_plan9b(TransCondGaussianDiffusion_plan9a):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

    def build_traj_reduce_weight(self):
        re_weight = torch.FloatTensor([i+1 for i in range(Configs.horizon)]).to(Configs.device).reshape(1,1,Configs.horizon)
        traj_reduce_weight = 1/re_weight
        return traj_reduce_weight #/torch.sum(traj_reduce_weight, -1)


class TransCondGaussianDiffusion_plan9bf(TransCondGaussianDiffusion_plan9af):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

    def build_traj_reduce_weight(self):
        re_weight = torch.FloatTensor([i+1 for i in range(Configs.horizon-1)]).to(Configs.device).reshape(1,1,Configs.horizon-1)
        traj_reduce_weight = 1/re_weight
        return  traj_reduce_weight#/torch.sum(traj_reduce_weight, -1)


class TransCondGaussianDiffusion_plan9bfn(TransCondGaussianDiffusion_plan9af):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

    def build_traj_reduce_weight(self):
        re_weight = torch.FloatTensor([i+1 for i in range(Configs.horizon-1)]).to(Configs.device).reshape(1,1,Configs.horizon-1)
        traj_reduce_weight = 1/re_weight
        return  traj_reduce_weight/torch.sum(traj_reduce_weight, -1)


class TransCondGaussianDiffusion_plan9c(TransCondGaussianDiffusion_plan9a):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)


class TransCondGaussianDiffusion_plan9d(TransCondGaussianDiffusion_plan9a):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

    def p_losses(self, x_start, cond, positives, negatives, positives_vals, negatives_vals,  history,  t  ):

        noise = torch.randn_like(x_start)


        # history_embd = self.encoder(history_obs, history_act, mask = self.build_mask())

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)


        # if Configs.reduce_method == "mean":
        reduc_weight_posi = torch.ones( x_start.shape[0], Configs.subbatchsize ).to(Configs.device) / Configs.subbatchsize
        reduc_weight_nega = torch.ones( x_start.shape[0], Configs.subbatchsize ).to(Configs.device) / Configs.subbatchsize

        x_recon_embd  = self.contrastive_embd_layer( x_recon[:, :, self.action_dim:] )
        positive_embd = self.contrastive_embd_layer( positives )#.mean(1))
        negative_embd = self.contrastive_embd_layer( negatives )



        cons_loss = self.contrastive_loss_fn(x_recon_embd, positive_embd, negative_embd,  reduc_weight_posi = reduc_weight_posi, reduc_weight_nega = reduc_weight_nega , traj_reduce_weight = self.traj_reduce_weight  )

        info_pack =  {'diff_loss': loss, 'cons_loss': cons_loss}
        info_pack.update(info)

        loss_all = cons_loss*Configs.contrastweigth + loss

        return loss_all, info_pack

TransCondGaussianDiffusion_plan9dfn = TransCondGaussianDiffusion_plan9bfn
TransCondGaussianDiffusion_plan9df = TransCondGaussianDiffusion_plan9b

class TransCondGaussianDiffusion_plan10(TransCondGaussianDiffusion_plan9f):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan10a(TransCondGaussianDiffusion_plan9af):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan10b(TransCondGaussianDiffusion_plan9bf):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan10d(TransCondGaussianDiffusion_plan9df):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan11(TransCondGaussianDiffusion_plan9b):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan11f(TransCondGaussianDiffusion_plan9bf):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class ContrastiveDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

        self.contrastive_loss_fn = InfoNCE(temperature =  Configs.temperature, negative_mode = "paired") if Configs.conembver=="traj" else  soft_info_nce_traj
        self.inputdim =  self.horizon*self.transition_dim  if Configs.conembver=="traj" else   self.observation_dim
        # conembver
        self.contrastive_embd_layer = nn.Sequential(
                nn.Linear(self.inputdim , Configs.contrastiveembd),
                eval(f"nn.{Configs.act}()"),
            ) if Configs.contrastiveembd > 0 else nn.Identity()
        
        self.traj_reduce_weight = self.build_traj_reduce_weight()
        
    def build_traj_reduce_weight(self):
        re_weight = torch.FloatTensor([i+1 for i in range(Configs.horizon)]).to(Configs.device).reshape(1,1,Configs.horizon)
        traj_reduce_weight = 1/re_weight
        return traj_reduce_weight 
    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device

        sample_fn = eval(f"n_step_guided_p_sample_{Configs.branch}")
        
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()

        x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, **sample_kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, positive, negative, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        reduc_weight_posi = torch.ones( x_start.shape[0], Configs.subbatchsize ).to(Configs.device) / Configs.subbatchsize
        reduc_weight_nega = torch.ones( x_start.shape[0], Configs.subbatchsize ).to(Configs.device) / Configs.subbatchsize
        
        if Configs.conembver == "traj":
            x_recon_embd = self.contrastive_embd_layer(einops.rearrange(x_recon, 'b h e -> b (h e)'))
            positive_embd = self.contrastive_embd_layer(einops.rearrange(positive, 'b bs h e -> b bs (h e)').mean(1))
            negative_embd = self.contrastive_embd_layer(einops.rearrange(negative, 'b bs h e -> b bs (h e)'))
            cons_loss = self.contrastive_loss_fn(x_recon_embd, positive_embd, negative_embd)
        else:
            x_recon_embd  = self.contrastive_embd_layer( x_recon[:, :, self.action_dim:] )
            positive_embd = self.contrastive_embd_layer( positive[:,:,:,self.action_dim:] )#.mean(1))
            negative_embd = self.contrastive_embd_layer( negative[:,:,:,self.action_dim:] )
            cons_loss = self.contrastive_loss_fn(x_recon_embd, positive_embd, negative_embd,  reduc_weight_posi = reduc_weight_posi, reduc_weight_nega = reduc_weight_nega , traj_reduce_weight = self.traj_reduce_weight  )
        
        info_pack =  {'diff_loss': loss, 'cons_loss': cons_loss}
        info_pack.update(info)

        return  loss+cons_loss*Configs.contrastweigth, info

    def loss(self, x, cond, positive, negative):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, positive, negative, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)
    
class TransCondGaussianDiffusion_plan12(ContrastiveDiffusion):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan14a1(TransCondGaussianDiffusion_plan9b):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan14a2(TransCondGaussianDiffusion_plan9b):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan14b(TransCondGaussianDiffusion_plan9b):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan15a(TransCondGaussianDiffusion_plan9b):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)
    
    def build_con_mask():

        return
    def p_losses(self, x_start, cond, positives, negatives, positives_vals, negatives_vals,  history,  t  ):

        noise = torch.randn_like(x_start)


        # history_embd = self.encoder(history_obs, history_act, mask = self.build_mask())

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        # 构建mask，将一定比例的mask位置置为1，其他为0
        rand_m = torch.rand(x_start.shape[0], Configs.subbatchsize, Configs.horizon, device = Configs.device)
        if x_start.shape[0] != Configs.batch_size:
            print(f"Warning: incomplete batch found (shape @ {x_start.shape})")
        mask = rand_m > Configs.contrastratio
        # if Configs.reduce_method == "mean":
        reduc_weight_posi = torch.ones( x_start.shape[0], Configs.subbatchsize ).to(Configs.device) / Configs.subbatchsize
        reduc_weight_nega = torch.ones( x_start.shape[0], Configs.subbatchsize ).to(Configs.device) / Configs.subbatchsize

        assert Configs.conembver == "state"
        if Configs.conembver == "traj":
            x_recon_embd = self.contrastive_embd_layer(einops.rearrange(x_recon[:, :, self.action_dim:], 'b h e -> b (h e)'))
            positive_embd = self.contrastive_embd_layer(einops.rearrange(positives, 'b bs h e -> b bs (h e)').mean(1))
            negative_embd = self.contrastive_embd_layer(einops.rearrange(negatives, 'b bs h e -> b bs (h e)'))
            cons_loss = self.contrastive_loss_fn(x_recon_embd, positive_embd, negative_embd)
        else:
            x_recon_embd  = self.contrastive_embd_layer( x_recon[:, :, self.action_dim:] )
            positive_embd = self.contrastive_embd_layer( positives )#.mean(1))
            negative_embd = self.contrastive_embd_layer( negatives )
            cons_loss = self.contrastive_loss_fn(x_recon_embd, positive_embd, negative_embd,  reduc_weight_posi = reduc_weight_posi, reduc_weight_nega = reduc_weight_nega , traj_reduce_weight = self.traj_reduce_weight, mask = mask  )



        # cons_loss = self.contrastive_loss_fn(x_recon_embd, positive_embd, negative_embd,  reduc_weight_posi = reduc_weight_posi, reduc_weight_nega = reduc_weight_nega , traj_reduce_weight = self.traj_reduce_weight  )

        info_pack =  {'diff_loss': loss, 'cons_loss': cons_loss}
        info_pack.update(info)

        loss_all = cons_loss*Configs.contrastweigth + loss

        return loss_all, info_pack



class TransCondGaussianDiffusion_plan14d1(TransCondGaussianDiffusion_plan14b):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)
    
    def p_losses(self, x_start, cond, positives, negatives, positives_vals, negatives_vals,  history,  t  ):

        # stats = x_start[:,:, self.action_dim:]
        # stat_emb = self.state_pre_encoder(stats)
        # cond_0 = self.state_pre_encoder(cond[0])
        noise = torch.randn_like(x_start)


        # history_embd = self.encoder(history_obs, history_act, mask = self.build_mask())

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        info_pack =  {'diff_loss': loss}
        info_pack.update(info)

        loss_all = loss

        return loss_all, info_pack
    

class TransCondGaussianDiffusion_plan14d2(TransCondGaussianDiffusion_plan14b):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)
    
    def p_losses(self, x_start, cond, positives, negatives, positives_vals, negatives_vals,  history,  t  ):

        # stats = x_start[:,:, self.action_dim:]
        # stat_emb = self.state_pre_encoder(stats)
        # cond_0 = self.state_pre_encoder(cond[0])
        noise = torch.randn_like(x_start)


        # history_embd = self.encoder(history_obs, history_act, mask = self.build_mask())

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)


        info_pack =  {'diff_loss': loss}
        info_pack.update(info)

        loss_all = loss

        return loss_all, info_pack
    

class TransCondGaussianDiffusion_plan14d3(TransCondGaussianDiffusion_plan14d2):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)



# use positive samples only
class TransCondGaussianDiffusion_plan14e(TransCondGaussianDiffusion_plan14d1):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)
    
    def p_losses(self, x_start, cond,  rewards,   t  ):

        # stats = x_start[:,:, self.action_dim:]
        # stat_emb = self.state_pre_encoder(stats)
        # cond_0 = self.state_pre_encoder(cond[0])
        noise = torch.randn_like(x_start)


        # history_embd = self.encoder(history_obs, history_act, mask = self.build_mask())

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        info_pack =  {'diff_loss': loss}
        info_pack.update(info)

        loss_all = loss

        return loss_all, info_pack
    

    

class TransCondGaussianDiffusion_plan14g(GaussianDiffusion):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)
    
    def p_losses(self, x_start, cond,  rewards,   t  ):

        # stats = x_start[:,:, self.action_dim:]
        # stat_emb = self.state_pre_encoder(stats)
        # cond_0 = self.state_pre_encoder(cond[0])
        noise = torch.randn_like(x_start)


        # history_embd = self.encoder(history_obs, history_act, mask = self.build_mask())

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        info_pack =  {'diff_loss': loss}
        info_pack.update(info)

        loss_all = loss

        return loss_all, info_pack
    
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None
        sample_fn = eval(f"n_step_guided_p_sample_{Configs.branch}")
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            x = apply_conditioning(x, cond, self.action_dim)

            progress.update({'t': i, 'vmin': values.min().item(), 'vmax': values.max().item()})
            if return_chain: chain.append(x)

        progress.stamp()

        x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain)
    
class TransCondGaussianDiffusion_plan14f1(TransCondGaussianDiffusion_plan14b):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

    
class TransCondGaussianDiffusion_plan14f2(TransCondGaussianDiffusion_plan14b):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan14bf(TransCondGaussianDiffusion_plan14b):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)

class TransCondGaussianDiffusion_plan15(TransCondGaussianDiffusion_plan14bf):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)



class TransCondGaussianDiffusion_plan15_diffuser(GaussianDiffusion):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)        

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)  # ?

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        info["Total_loss"] =  loss
        info["Diff_loss"] =  loss

        return loss, info
    

class TransCondGaussianDiffusion_plan15bf(TransCondGaussianDiffusion_plan14bf):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)        

    def p_losses(self, x_start, cond, positives, negatives, positives_vals, negatives_vals,  history,  t  ):

        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)


        # if Configs.reduce_method == "mean":
        reduc_weight_posi = torch.ones( x_start.shape[0], Configs.subbatchsize ).to(Configs.device) / Configs.subbatchsize
        reduc_weight_nega = torch.ones( x_start.shape[0], Configs.subbatchsize ).to(Configs.device) / Configs.subbatchsize

        # always state level
        if Configs.conembver == "traj":  
            x_recon_embd = self.contrastive_embd_layer(einops.rearrange(x_recon[:, :, self.action_dim:], 'b h e -> b (h e)'))
            positive_embd = self.contrastive_embd_layer(einops.rearrange(positives, 'b bs h e -> b bs (h e)').mean(1))
            negative_embd = self.contrastive_embd_layer(einops.rearrange(negatives, 'b bs h e -> b bs (h e)'))
            cons_loss = self.contrastive_loss_fn(x_recon_embd, positive_embd, negative_embd)
        else:
            x_recon_embd  = self.contrastive_embd_layer( x_recon[:, :, self.action_dim:] )
            positive_embd = self.contrastive_embd_layer( positives )#.mean(1))
            negative_embd = self.contrastive_embd_layer( negatives )
            cons_loss = self.contrastive_loss_fn(x_recon_embd, positive_embd, negative_embd,  reduc_weight_posi = reduc_weight_posi, reduc_weight_nega = reduc_weight_nega , traj_reduce_weight = self.traj_reduce_weight  )



        # cons_loss = self.contrastive_loss_fn(x_recon_embd, positive_embd, negative_embd,  reduc_weight_posi = reduc_weight_posi, reduc_weight_nega = reduc_weight_nega , traj_reduce_weight = self.traj_reduce_weight  )

        

        loss_all = cons_loss*Configs.contrastweigth + loss


        info_pack =  {'Diff_loss': loss, 'Total_loss': loss_all, "Contrast_Loss": cons_loss}
        info_pack.update(info)


        return loss_all, info_pack

class TransCondGaussianDiffusion_plan15f1(TransCondGaussianDiffusion_plan14bf):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)        


class TransCondGaussianDiffusion_plan15f_diffuser(TransCondGaussianDiffusion_plan15_diffuser):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)      


class TransCondGaussianDiffusion_plan15fbf(TransCondGaussianDiffusion_plan15bf):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)        


class TransCondGaussianDiffusion_plan15ff1(TransCondGaussianDiffusion_plan15f1):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)        



class CDiffusion_plan1_diffuser(GaussianDiffusion):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)        

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)  # ?

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)
        info['TotalLoss'] = loss
        info['DiffLoss'] = loss

        return loss, info
    
class CDiffusion_plan1_hard(TransCondGaussianDiffusion_plan14bf):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)        
        self.contrastive_loss_fn = self.infonce

    def infonce(self, query, positive_key, negative_keys, reduc_weight_posi, reduc_weight_nega, traj_reduce_weight, mask = None ):
        temperature = Configs.temperature
        num_posi = positive_key.shape[1]
        cos_positive = F.cosine_similarity(  query.unsqueeze(1) , positive_key  , -1  , eps= 1e-12 )
        exp_pos = torch.exp(cos_positive/temperature) 
        exp_pos = torch.sum(  exp_pos*traj_reduce_weight , -1 )
        numerator = torch.sum(exp_pos*reduc_weight_posi , -1 )/num_posi   # top  这里的sum是将多个正样本loss融合

        if Configs.lowerbound == 0:
            return torch.mean(-torch.log(  numerator ))

        cos_negative = F.cosine_similarity(  query.unsqueeze(1) , negative_keys , -1  , eps= 1e-12  )
        exp_nega = torch.exp(cos_negative/temperature ) 
        exp_nega = torch.sum(  exp_nega*traj_reduce_weight , -1 )
        denominator  = torch.sum(exp_nega*reduc_weight_nega , -1 )
        
        return torch.mean(-torch.log(  numerator / denominator ))

class CDiffusion_plan1_hardnce(CDiffusion_plan1_hard):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)        
        self.contrastive_loss_fn = self.soft_info_nce_traj_fix

    def soft_info_nce_traj_fix(self, query, positive_key, negative_keys, reduc_weight_posi, reduc_weight_nega, traj_reduce_weight  ):

        temperature = Configs.temperature

        num_posi = positive_key.shape[1]
        cos_positive = F.cosine_similarity(  query.unsqueeze(1) , positive_key  , -1  , eps= 1e-12 )
        cos_negative = F.cosine_similarity(  query.unsqueeze(1) , negative_keys , -1  , eps= 1e-12  )
        exp_pos = torch.exp(cos_positive/temperature) 
        exp_nega = torch.exp(cos_negative/temperature ) 

        exp_pos = torch.sum(  exp_pos*traj_reduce_weight , -1 )
        exp_nega = torch.sum(  exp_nega*traj_reduce_weight , -1 )

        numerator = torch.sum(exp_pos*reduc_weight_posi , -1 )/num_posi  
        denominator  = torch.sum(exp_nega*reduc_weight_nega , -1 ) + numerator


        return torch.mean(-torch.log(  numerator / denominator ))

class CDiffusion_plan2_hard(CDiffusion_plan1_hard):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)        

class CDiffusion_plan3(CDiffusion_plan2_hard):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)
    



class CDiffusion_plan1_nomix(CDiffusion_plan1_hard):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)        
class CDiffusion_plan2_nomix(CDiffusion_plan2_hard):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000, loss_type='l1', clip_denoised=False, predict_epsilon=True, action_weight=1, loss_discount=1, loss_weights=None):
        super().__init__(model, horizon, observation_dim, action_dim, n_timesteps, loss_type, clip_denoised, predict_epsilon, action_weight, loss_discount, loss_weights)        
