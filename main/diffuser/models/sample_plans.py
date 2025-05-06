import torch
from diffuser.sampling.functions import n_step_guided_p_sample
from diffuser.models.helpers import (
    extract,
    apply_conditioning,
)

from config.locomotion_config import Configs

@torch.no_grad()
def n_step_guided_p_sample_plan3a(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,history = None, step = 0, history_obs = None, history_act = None
):
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            y, grad = guide.gradients(x, cond, t)

        if scale_grad_by_std:
            grad = model_var * grad

        grad[t < t_stopgrad] = 0

        x = x + scale * grad
        x = apply_conditioning(x, cond, model.action_dim)

    if n_guide_steps == 0:
        y, grad = guide.gradients(x, cond, t)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t, history_obs = history_obs, history_act = history_act, step = step)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y



@torch.no_grad()
def n_step_guided_p_sample_plan9(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,
):
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            y, grad = guide.gradients(x, cond, t)

        if scale_grad_by_std:
            grad = model_var * grad

        grad[t < t_stopgrad] = 0

        x = x + scale * grad
        x = apply_conditioning(x, cond, model.action_dim)

    if n_guide_steps == 0:
        with torch.enable_grad():
            y, grad = guide.gradients(x, cond, t)
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y

n_step_guided_p_sample_diffuser = \
n_step_guided_p_sample_plan14bf = \
n_step_guided_p_sample_plan14g = \
n_step_guided_p_sample_plan14f2 = \
n_step_guided_p_sample_plan14f1 = \
n_step_guided_p_sample_plan14e = \
n_step_guided_p_sample_plan14d3 = \
n_step_guided_p_sample_plan14d1 = \
n_step_guided_p_sample_plan14d2 = \
n_step_guided_p_sample_plan14b = \
n_step_guided_p_sample_plan15a = \
n_step_guided_p_sample_plan14a2 = \
n_step_guided_p_sample_plan14a1 = \
n_step_guided_p_sample_plan12 = \
n_step_guided_p_sample_plan11 = \
n_step_guided_p_sample_plan10d = \
n_step_guided_p_sample_plan10b = \
n_step_guided_p_sample_plan10a = \
n_step_guided_p_sample_plan10 = \
n_step_guided_p_sample_plan9f = \
n_step_guided_p_sample_plan9fn = \
n_step_guided_p_sample_plan9d = \
n_step_guided_p_sample_plan9df = \
n_step_guided_p_sample_plan9dfn = \
n_step_guided_p_sample_plan9afn = \
n_step_guided_p_sample_plan9bfn = \
n_step_guided_p_sample_plan9cfn = \
n_step_guided_p_sample_plan9af = \
n_step_guided_p_sample_plan9bf = \
n_step_guided_p_sample_plan9cf = \
n_step_guided_p_sample_plan9a = \
n_step_guided_p_sample_plan9b = \
n_step_guided_p_sample_plan9c = \
n_step_guided_p_sample_plan9


n_step_guided_p_sample_plan8a = \
n_step_guided_p_sample_plan7dNorm = \
n_step_guided_p_sample_plan7dM = \
n_step_guided_p_sample_plan7dN = \
n_step_guided_p_sample_plan7dS = \
n_step_guided_p_sample_plan7aNorm = \
n_step_guided_p_sample_plan7aM = \
n_step_guided_p_sample_plan7aN = \
n_step_guided_p_sample_plan7aS = \
n_step_guided_p_sample_plan7 = \
n_step_guided_p_sample_plan6F1Norm = \
n_step_guided_p_sample_plan6T1Norm = \
n_step_guided_p_sample_plan6T1S = \
n_step_guided_p_sample_plan6T1N = \
n_step_guided_p_sample_plan6T1M = \
n_step_guided_p_sample_plan6F1S = \
n_step_guided_p_sample_plan6F1N = \
n_step_guided_p_sample_plan6F1M = \
n_step_guided_p_sample_plan6F2Norm = \
n_step_guided_p_sample_plan6T2Norm = \
n_step_guided_p_sample_plan6T2S = \
n_step_guided_p_sample_plan6T2N = \
n_step_guided_p_sample_plan6T2M = \
n_step_guided_p_sample_plan6F2S = \
n_step_guided_p_sample_plan6F2N = \
n_step_guided_p_sample_plan6F2M = \
n_step_guided_p_sample_plan6c = \
    n_step_guided_p_sample_plan6b = \
        n_step_guided_p_sample_plan6a =  \
            n_step_guided_p_sample_plan6 = \
                n_step_guided_p_sample_plan5a = n_step_guided_p_sample_plan3a