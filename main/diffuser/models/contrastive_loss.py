import torch
import torch.nn.functional as F
from torch import nn
from config.locomotion_config import Configs


class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.

    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113

    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.

    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.

    Returns:
         Value of the InfoNCE Loss.

     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    # if positive_key.dim() != 2:
    #     raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)


def transpose(x):
    return x.transpose(-2, -1)


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]



import einops

# class SoftInfoNCE(nn.Module):

#     def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired', reweight = 'consistent'):
#         super().__init__()
#         self.temperature = temperature
#         self.reduction = reduction
#         self.negative_mode = negative_mode

#         self.reweight = reweight  #描述采取什么方式给正样本/负样本赋权

#     def forward(self, query, positive_key, negative_keys=None):
#         return info_nce(query, positive_key, negative_keys,
#                         temperature=self.temperature,
#                         reduction=self.reduction,
#                         negative_mode=self.negative_mode)


def soft_info_nce_traj(query, positive_key, negative_keys, reduc_weight_posi, reduc_weight_nega, traj_reduce_weight, mask = None ):

    temperature = Configs.temperature

    num_posi = positive_key.shape[1]

    

    cos_positive = F.cosine_similarity(  query.unsqueeze(1) , positive_key  , -1  , eps= 1e-12 )
    cos_negative = F.cosine_similarity(  query.unsqueeze(1) , negative_keys , -1  , eps= 1e-12  )
    
    if mask is not None:
        cos_positive[mask] = 1
        cos_negative[mask] = -1 
        
    exp_pos = torch.exp(cos_positive/temperature) 
    exp_nega = torch.exp(cos_negative/temperature ) 

    # 在这里融合traj内的正负样本，给不同权重。理由：越向后的预测结果越不准确
    # 期望的traj_reduce_weight： 1,1,horizon -> batch subbatch horizon 1(omit)
    # traj_reduce_weight = torch.ones(   [1,1,Configs.horizon]    , device=exp_nega.device )
    exp_pos = torch.sum(  exp_pos*traj_reduce_weight , -1 )
    exp_nega = torch.sum(  exp_nega*traj_reduce_weight , -1 )

    numerator = torch.sum(exp_pos*reduc_weight_posi , -1 )/num_posi   # top  这里的sum是将多个正样本loss融合
    denominator  = torch.sum(exp_nega*reduc_weight_nega , -1 )


    return torch.mean(-torch.log(  numerator / denominator ))


def soft_info_nce(query, positive_key, negative_keys, reduc_weight_posi, reduc_weight_nega ):

    temperature = Configs.temperature

    num_posi = positive_key.shape[1]

    cos_positive = F.cosine_similarity(  query.unsqueeze(1) , positive_key  , -1  , eps= 1e-12 )
    cos_negative = F.cosine_similarity(  query.unsqueeze(1) , negative_keys , -1  , eps= 1e-12  )
    exp_pos = torch.exp(cos_positive/temperature) 
    exp_nega = torch.exp(cos_negative/temperature ) 


    assert reduc_weight_posi.shape == exp_pos.shape

    numerator = torch.sum(exp_pos*reduc_weight_posi , -1 )/num_posi   # top
    denominator  = torch.sum(exp_nega*reduc_weight_nega , -1 )


    return torch.mean(-torch.log(  numerator / denominator ))