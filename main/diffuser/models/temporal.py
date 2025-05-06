import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb
from config.locomotion_config import Configs
from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LayerNorm,
    LinearAttention,
)


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=False,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, cond, time):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x


class TransCondTemporalUnet_diffuser(TemporalUnet):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)



class TransCondTemporalUnet_main(TemporalUnet):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)


        self.history_layer = nn.Sequential(
            nn.Linear(Configs.dmodel, 64 ),
            nn.Mish(),
            Rearrange("b history emb -> b (history emb)"),
            nn.Linear(Configs.history_length*64 , dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        ).to(Configs.device)
    
    def forward(self, x, cond, time, history_embd):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')
        history_aggregated = self.history_layer(history_embd)
        t = self.time_mlp(time)
        t = (t+history_aggregated)/2
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x

class ValueFunction(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out)
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4
        ##
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim_2, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        self.mid_down1 = Downsample1d(mid_dim_2)
        horizon = horizon // 2
        ##
        self.mid_block2 = ResidualTemporalBlock(mid_dim_2, mid_dim_3, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        self.mid_down2 = Downsample1d(mid_dim_3)
        horizon = horizon // 2
        ##
        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        ## mask out first conditioning timestep, since this is not sampled by the model
        # x[:, :, 0] = 0

        t = self.time_mlp(time)

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        ##
        x = self.mid_block1(x, t)
        x = self.mid_down1(x)
        ##
        x = self.mid_block2(x, t)
        x = self.mid_down2(x)
        ##
        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out



class TransCondTemporalUnet_plan1a(TemporalUnet):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):


        transition_dim *= 2


        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        
        # self.history_layer = nn.Sequential(
        #     nn.Linear(Configs.dmodel, 64 ),
        #     nn.Mish(),
        #     Rearrange("b history emb -> b (history emb)"),
        #     nn.Linear(Configs.history_length*64 , dim * 4),
        #     nn.Mish(),
        #     nn.Linear(dim * 4, dim),
        # ).to(Configs.device)

        self.reduce = nn.Linear(Configs.dmodel, Configs.transition_dim)
        self.testplan1_mask = torch.zeros( Configs.batch_size, Configs.horizon+Configs.history_length-1, Configs.observation_dim )

    def forward(self, x, cond, time, history_embd):

        # if Configs.test:
        #     return self.test(x, cond, time, history_embd)

        '''
            x : [ batch x horizon x transition ]
        '''
        history_embd_cond = history_embd[:,-Configs.horizon:, :]

        if Configs.test:
            if Configs.testplan == "1":
                pass
            if Configs.testplan == "2":
                history_embd_cond[:, -Configs.horizon:, :] = 0

        x = einops.rearrange(x, 'b h t -> b t h')
        # history_aggregated = self.history_layer(history_embd)

        history_embd_cond = self.reduce(history_embd_cond)
        x = torch.cat( [x, einops.rearrange(history_embd_cond, 'b h t -> b t h')] , 1 )
        t = self.time_mlp(time)
        # t = (t+history_aggregated)/2
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')[:,:,:Configs.transition_dim]
        return x


class TransCondTemporalUnet_plan1b(TemporalUnet):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):


        # transition_dim *= 2


        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.reduce = nn.Linear(Configs.dmodel, Configs.transition_dim)
        self.reduce2 = nn.Linear(Configs.transition_dim*2, Configs.transition_dim)

    def forward(self, x, cond, time, history_embd):
        '''
            x : [ batch x horizon x transition ]
        '''
        history_embd_cond = history_embd[:,-Configs.horizon:, :]

        x = einops.rearrange(x, 'b h t -> b t h')
        # history_aggregated = self.history_layer(history_embd)

        history_embd_cond = self.reduce(history_embd_cond)
        x = torch.cat( [x, einops.rearrange(history_embd_cond, 'b h t -> b t h')] , 1 )


        x = self.reduce2(x)

        t = self.time_mlp(time)
        # t = (t+history_aggregated)/2
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')[:,:,:Configs.transition_dim]
        return x


class TransCondTemporalUnet_plan2a(TransCondTemporalUnet_plan1a):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)


class TransCondTemporalUnet_plan3a(TransCondTemporalUnet_plan1a):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)


class TransCondTemporalUnet_plan5a(TransCondTemporalUnet_plan1b):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

    def forward(self, x, cond, time, history_embd):
        '''
            x : [ batch x horizon x transition ]
        '''
        history_embd_cond = history_embd[:,-Configs.horizon, :] 
        

        history_embd_cond = self.reduce(history_embd_cond)
        comb = torch.cat([x[:,0,:],  history_embd_cond ], -1)
        cond = self.reduce2(comb)
        x[:, 0, :] = cond.clone()

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        # t = (t+history_aggregated)/2
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')[:,:,:Configs.transition_dim]
        return x
    
class TransCondTemporalUnet_plan6(TransCondTemporalUnet_plan1b):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.state_embedding = nn.Sequential(
            nn.Linear(cond_dim, Configs.dimenc),
            nn.Linear(Configs.dimenc, cond_dim),
        )
        
        self.action_embedding = nn.Sequential(
            nn.Linear(transition_dim-cond_dim, Configs.dimenc),
            nn.Linear(Configs.dimenc, transition_dim-cond_dim),
        )



    def forward(self, x, cond, time, bert, mask, history_obs, history_act):
        '''
            x : [ batch x horizon x transition ]
        '''
        # history_embd_cond = history_embd[:,-Configs.horizon, :] 
        history_obs_emb = self.state_embedding(history_obs)
        history_act_emb = self.action_embedding(history_act)
        history_embd = bert(history_obs_emb, history_act_emb, mask)  # plan6是state做score， action做value
        history_embd_cond = history_embd[:,-Configs.horizon, :]  # 取出t时刻做完crossattention的condition
        history_embd_cond = self.reduce(history_embd_cond)


        # 重构x为embedding
        states = x[:, :, Configs.action_dim:]
        actions = x[:, :, :Configs.action_dim]

        states_embd = self.state_embedding(states)
        actions_embd = self.action_embedding(actions)

        # plan6作为对照实验，不添加历史信息
        x = torch.cat([ actions_embd, states_embd   ], -1)  
        # comb = torch.cat([x[:,0,:],  history_embd_cond ], -1)
        # cond = self.reduce2(comb)   # 将condition融合进去
        # x[:, 0, :] = cond.clone()

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        # t = (t+history_aggregated)/2
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')[:,:,:Configs.transition_dim]
        return x
    

# 双层的各个
class TransCondTemporalUnet_plan6T2S(TransCondTemporalUnet_plan6):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.state_embedding = nn.Sequential(
            nn.Linear(cond_dim, Configs.dimenc, bias = True),
            nn.Sigmoid(),
            nn.Dropout(Configs.tdropout),
            nn.Linear(Configs.dimenc, cond_dim, bias = True),
            nn.Sigmoid(),
            nn.Dropout(Configs.tdropout),
        )
        
        self.action_embedding = nn.Sequential(
            nn.Linear(transition_dim-cond_dim, Configs.dimenc, bias = True),
            nn.Sigmoid(),
            nn.Dropout(Configs.tdropout),
            nn.Linear(Configs.dimenc, transition_dim-cond_dim, bias = True),
            nn.Sigmoid(),
            nn.Dropout(Configs.tdropout),
        )


class TransCondTemporalUnet_plan6T2N(TransCondTemporalUnet_plan6):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.state_embedding = nn.Sequential(
            nn.Linear(cond_dim, Configs.dimenc, bias = True),
            # nn.Sigmoid(),
            nn.Linear(Configs.dimenc, cond_dim, bias = True),
            # nn.Sigmoid(),
        )
        
        self.action_embedding = nn.Sequential(
            nn.Linear(transition_dim-cond_dim, Configs.dimenc, bias = True),
            # nn.Sigmoid(),
            nn.Linear(Configs.dimenc, transition_dim-cond_dim, bias = True),
            # nn.Sigmoid(),
        )

class TransCondTemporalUnet_plan6T2M(TransCondTemporalUnet_plan6):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.state_embedding = nn.Sequential(
            nn.Linear(cond_dim, Configs.dimenc, bias = True),
            nn.Mish(),
            nn.Dropout(Configs.tdropout),
            nn.Linear(Configs.dimenc, cond_dim, bias = True),
            nn.Mish(),
            nn.Dropout(Configs.tdropout),
        )
        
        self.action_embedding = nn.Sequential(
            nn.Linear(transition_dim-cond_dim, Configs.dimenc, bias = True),
            nn.Mish(),
            nn.Dropout(Configs.tdropout),
            nn.Linear(Configs.dimenc, transition_dim-cond_dim, bias = True),
            nn.Mish(),
            nn.Dropout(Configs.tdropout),
        )




class TransCondTemporalUnet_plan6F2S(TransCondTemporalUnet_plan6):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.state_embedding = nn.Sequential(
            nn.Linear(cond_dim, Configs.dimenc, bias = False),
            nn.Sigmoid(),
            nn.Linear(Configs.dimenc, cond_dim, bias = False),
            nn.Sigmoid(),
        )
        
        self.action_embedding = nn.Sequential(
            nn.Linear(transition_dim-cond_dim, Configs.dimenc, bias = False),
            nn.Sigmoid(),
            nn.Linear(Configs.dimenc, transition_dim-cond_dim, bias = False),
            nn.Sigmoid(),
        )


class TransCondTemporalUnet_plan6F2N(TransCondTemporalUnet_plan6):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.state_embedding = nn.Sequential(
            nn.Linear(cond_dim, Configs.dimenc, bias = False),
            # nn.Sigmoid(),
            nn.Linear(Configs.dimenc, cond_dim, bias = False),
            # nn.Sigmoid(),
        )
        
        self.action_embedding = nn.Sequential(
            nn.Linear(transition_dim-cond_dim, Configs.dimenc, bias = False),
            # nn.Sigmoid(),
            nn.Linear(Configs.dimenc, transition_dim-cond_dim, bias = False),
            # nn.Sigmoid(),
        )

class TransCondTemporalUnet_plan6F2M(TransCondTemporalUnet_plan6):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.state_embedding = nn.Sequential(
            nn.Linear(cond_dim, Configs.dimenc, bias = False),
            nn.Mish(),
            nn.Linear(Configs.dimenc, cond_dim, bias = False),
            nn.Mish(),
        )
        
        self.action_embedding = nn.Sequential(
            nn.Linear(transition_dim-cond_dim, Configs.dimenc, bias = False),
            nn.Mish(),
            nn.Linear(Configs.dimenc, transition_dim-cond_dim, bias = False),
            nn.Mish(),
        )

class TransCondTemporalUnet_plan6F2Norm(TransCondTemporalUnet_plan6):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.state_embedding = nn.Sequential(
            nn.Linear(cond_dim, Configs.dimenc, bias = False),
            nn.LayerNorm( Configs.dimenc),
            nn.Linear(Configs.dimenc, cond_dim, bias = False),
            nn.LayerNorm(cond_dim),
        )
        
        self.action_embedding = nn.Sequential(
            nn.Linear(transition_dim-cond_dim, Configs.dimenc, bias = False),
            nn.LayerNorm(Configs.dimenc),
            nn.Linear(Configs.dimenc, transition_dim-cond_dim, bias = False),
            nn.LayerNorm(transition_dim-cond_dim),
        )

class TransCondTemporalUnet_plan6T2Norm(TransCondTemporalUnet_plan6):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.state_embedding = nn.Sequential(
            nn.Linear(cond_dim, Configs.dimenc, bias = True),
            nn.LayerNorm( Configs.dimenc),
            nn.Linear(Configs.dimenc, cond_dim, bias = True),
            nn.LayerNorm(cond_dim),
        )
        
        self.action_embedding = nn.Sequential(
            nn.Linear(transition_dim-cond_dim, Configs.dimenc, bias = True),
            nn.LayerNorm( Configs.dimenc),
            nn.Linear(Configs.dimenc, transition_dim-cond_dim, bias = True),
            nn.LayerNorm(transition_dim-cond_dim),
        )



# 单层的各个
class TransCondTemporalUnet_plan6T1S(TransCondTemporalUnet_plan6):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.state_embedding = nn.Sequential(
            nn.Linear(cond_dim, cond_dim, bias = True),
            nn.Sigmoid(),
        )
        
        self.action_embedding = nn.Sequential(
            nn.Linear(transition_dim-cond_dim, transition_dim-cond_dim, bias = True),
            nn.Sigmoid(),
        )


class TransCondTemporalUnet_plan6T1N(TransCondTemporalUnet_plan6):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.state_embedding = nn.Sequential(
            nn.Linear(cond_dim,cond_dim, bias = True),
            # nn.Sigmoid(),
        )
        
        self.action_embedding = nn.Sequential(
            nn.Linear(transition_dim-cond_dim, transition_dim-cond_dim, bias = True),
            # nn.Sigmoid(),
        )

class TransCondTemporalUnet_plan6T1M(TransCondTemporalUnet_plan6):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.state_embedding = nn.Sequential(
            nn.Linear(cond_dim, cond_dim, bias = True),
            nn.Mish(),
        )
        
        self.action_embedding = nn.Sequential(
            nn.Linear(transition_dim-cond_dim, transition_dim-cond_dim, bias = True),
            nn.Mish(),
        )




class TransCondTemporalUnet_plan6F1S(TransCondTemporalUnet_plan6):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.state_embedding = nn.Sequential(
            nn.Linear(cond_dim, cond_dim, bias = False),
            nn.Sigmoid(),
        )
        
        self.action_embedding = nn.Sequential(
            nn.Linear(transition_dim-cond_dim, transition_dim-cond_dim, bias = False),
            nn.Sigmoid(),
        )


class TransCondTemporalUnet_plan6F1N(TransCondTemporalUnet_plan6):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.state_embedding = nn.Sequential(
            nn.Linear(cond_dim, cond_dim, bias = False),
            # nn.Sigmoid(),
        )
        
        self.action_embedding = nn.Sequential(
            nn.Linear(transition_dim-cond_dim, transition_dim-cond_dim, bias = False),
            # nn.Sigmoid(),
        )

class TransCondTemporalUnet_plan6F1M(TransCondTemporalUnet_plan6):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.state_embedding = nn.Sequential(
            nn.Linear(cond_dim, cond_dim, bias = False),
            nn.Mish(),
        )
        
        self.action_embedding = nn.Sequential(
            nn.Linear(transition_dim-cond_dim, transition_dim-cond_dim, bias = False),
            nn.Mish(),
        )

class TransCondTemporalUnet_plan6F1Norm(TransCondTemporalUnet_plan6):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.state_embedding = nn.Sequential(
            nn.Linear(cond_dim,cond_dim, bias = False),
            nn.LayerNorm(cond_dim),
        )
        
        self.action_embedding = nn.Sequential(
            nn.Linear(transition_dim-cond_dim, transition_dim-cond_dim, bias = False),
            nn.LayerNorm(transition_dim-cond_dim),
        )

class TransCondTemporalUnet_plan6T1Norm(TransCondTemporalUnet_plan6):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.state_embedding = nn.Sequential(
            nn.Linear(cond_dim, cond_dim, bias = True),
            nn.LayerNorm(cond_dim),
        )
        
        self.action_embedding = nn.Sequential(
            nn.Linear(transition_dim-cond_dim, transition_dim-cond_dim, bias = True),
            nn.LayerNorm(transition_dim-cond_dim),
        )


class TransCondTemporalUnet_plan6a(TransCondTemporalUnet_plan6):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

    def forward(self, x, cond, time, bert, mask, history_obs, history_act):
        '''
            x : [ batch x horizon x transition ]
        '''
        # history_embd_cond = history_embd[:,-Configs.horizon, :] 
        history_obs_emb = self.state_embedding(history_obs)
        history_act_emb = self.action_embedding(history_act)
        history_embd = bert(history_obs_emb,history_act_emb, mask)  # plan6是state做score， action做value
        history_embd_cond = history_embd[:,-Configs.horizon, :]  # 取出t时刻做完crossattention的condition
        history_embd_cond = self.reduce(history_embd_cond)


        # 重构x为embedding
        states = x[:, :, Configs.action_dim:]
        actions = x[:, :, :Configs.action_dim]

        states_embd = self.state_embedding(states)
        actions_embd = self.action_embedding(actions)

        # plan6作为对照实验，不添加历史信息,plan6a 和 6b需要将历史信息添加进去。
        x = torch.cat([ actions_embd, states_embd   ], -1)  
        
        comb = torch.cat([x[:,0,:],  history_embd_cond ], -1)
        cond = self.reduce2(comb)   # 将condition融合进去
        x[:, 0, :] = cond.clone()

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        # t = (t+history_aggregated)/2
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')[:,:,:Configs.transition_dim]
        return x



class TransCondTemporalUnet_plan6b(TransCondTemporalUnet_plan6a):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

    def forward(self, x, cond, time, bert, mask, history_obs, history_act):
        '''
            x : [ batch x horizon x transition ]
        '''
        # history_embd_cond = history_embd[:,-Configs.horizon, :] 
        history_obs_emb = self.state_embedding(history_obs)
        history_act_emb = self.action_embedding(history_act)
        history_embd = bert(history_obs_emb , mask)  # plan6是state做score， action做value
        history_embd_cond = history_embd[:,-Configs.horizon, :]  # 取出t时刻做完crossattention的condition
        history_embd_cond = self.reduce(history_embd_cond)


        # 重构x为embedding
        states = x[:, :, Configs.action_dim:]
        actions = x[:, :, :Configs.action_dim]

        states_embd = self.state_embedding(states)
        actions_embd = self.action_embedding(actions)

        # plan6作为对照实验，不添加历史信息,plan6a 和 6b需要将历史信息添加进去。
        x = torch.cat([ actions_embd, states_embd   ], -1)  
        comb = torch.cat([x[:,0,:],  history_embd_cond ], -1)
        cond = self.reduce2(comb)   # 将condition融合进去
        x[:, 0, :] = cond.clone()

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        # t = (t+history_aggregated)/2
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')[:,:,:Configs.transition_dim]
        return x


class TransCondTemporalUnet_plan6c(TransCondTemporalUnet_plan6a):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)
    
    def forward(self, x, cond, time, bert, mask, history_obs, history_act):
        '''
            x : [ batch x horizon x transition ]
        '''
        # history_embd_cond = history_embd[:,-Configs.horizon, :] 
        history_obs_emb = self.state_embedding(history_obs)
        history_act_emb = self.action_embedding(history_act)
        history_embd = bert( torch.cat(  [history_act_emb,history_obs_emb ] , -1 ), mask)  # plan6是state做score， action做value
        history_embd_cond = history_embd[:,-Configs.horizon, :]  # 取出t时刻做完crossattention的condition
        history_embd_cond = self.reduce(history_embd_cond)


        # 重构x为embedding
        states = x[:, :, Configs.action_dim:]
        actions = x[:, :, :Configs.action_dim]

        states_embd = self.state_embedding(states)
        actions_embd = self.action_embedding(actions)

        # plan6作为对照实验，不添加历史信息,plan6a 和 6b需要将历史信息添加进去。
        x = torch.cat([ actions_embd, states_embd   ], -1)  
        comb = torch.cat([x[:,0,:],  history_embd_cond ], -1)
        cond = self.reduce2(comb)   # 将condition融合进去
        x[:, 0, :] = cond.clone()

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        # t = (t+history_aggregated)/2
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')[:,:,:Configs.transition_dim]
        return x
    
    #TransCondTemporalUnet_plan6T1S

class TransCondTemporalUnet_plan7(TransCondTemporalUnet_plan6T1S):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)
    
    def forward(self, x, cond, time, bert, mask, history_obs, history_act):
        '''
            x : [ batch x horizon x transition ]
        '''
        # history_embd_cond = history_embd[:,-Configs.horizon, :] 
        history_obs_emb = self.state_embedding(history_obs)
        history_act_emb = self.action_embedding(history_act)
        history_embd = bert( torch.cat(  [history_act_emb,history_obs_emb ] , -1 ), mask)  # plan6是state做score， action做value
        history_embd_cond = history_embd[:,-Configs.horizon, :]  # 取出t时刻做完crossattention的condition
        history_embd_cond = self.reduce(history_embd_cond)


        # 重构x为embedding
        states = x[:, :, Configs.action_dim:]
        actions = x[:, :, :Configs.action_dim]

        states_embd = self.state_embedding(states)
        actions_embd = self.action_embedding(actions)

        # plan6作为对照实验，不添加历史信息,plan6a 和 6b需要将历史信息添加进去。
        x = torch.cat([ actions_embd, states_embd   ], -1)  
        comb = torch.cat([x[:,0,:],  history_embd_cond ], -1)
        cond = self.reduce2(comb)   # 将condition融合进去
        x[:, 0, :] = cond.clone()

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        # t = (t+history_aggregated)/2
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')[:,:,:Configs.transition_dim]
        return x
    
class TransCondTemporalUnet_plan7aN(TransCondTemporalUnet_plan6T1S): # s作为qk，s作v替换s0
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.reduce = nn.Linear(Configs.dmodel, Configs.observation_dim)
        self.reduce2 = nn.Linear(Configs.observation_dim*2, Configs.observation_dim)

    def forward(self, x, cond, time, bert, mask, history_obs, history_act):
        '''
            x : [ batch x horizon x transition ]
        '''
        # history_embd_cond = history_embd[:,-Configs.horizon, :] 
        history_obs_emb = self.state_embedding(history_obs)
        # history_act_emb = self.action_embedding(history_act)
        history_embd = bert( history_obs_emb , mask)  
        history_embd_cond = history_embd[:,-Configs.horizon, :] #
        history_embd_cond = self.reduce(history_embd_cond)


        # 重构x为embedding
        states = x[:, :, Configs.action_dim:]
        actions = x[:, :, :Configs.action_dim]

        states_embd = self.state_embedding(states)
        actions_embd = self.action_embedding(actions)

        # plan6作为对照实验，不添加历史信息,plan6a 和 6b需要将历史信息添加进去。
        x = torch.cat([ actions_embd, states_embd   ], -1)  

        comb = torch.cat([x[:,0, Configs.action_dim :],  history_embd_cond ], -1)

        cond = self.reduce2(comb)   # 将condition融合进去
        x[:, 0, Configs.action_dim:] = cond.clone()

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        # t = (t+history_aggregated)/2
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')[:,:,:Configs.transition_dim]
        return x

class TransCondTemporalUnet_plan7aS(TransCondTemporalUnet_plan7aN): # s作为qk，s作v替换s0
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.reduce = nn.Sequential(
            nn.Linear(Configs.dmodel, Configs.observation_dim),
            nn.Sigmoid(),
            nn.Dropout(Configs.tdropout),
        )
        self.reduce2 = nn.Sequential(
            nn.Linear(Configs.observation_dim*2, Configs.observation_dim),
            nn.Sigmoid(),
            nn.Dropout(Configs.tdropout),
        )
    
class TransCondTemporalUnet_plan7aM(TransCondTemporalUnet_plan7aN): # s作为qk，s作v替换s0
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.reduce = nn.Sequential(
            nn.Linear(Configs.dmodel, Configs.observation_dim),
            nn.Mish(),
        )
        self.reduce2 = nn.Sequential(
            nn.Linear(Configs.observation_dim*2, Configs.observation_dim),
            nn.Mish(),
        )
    

class TransCondTemporalUnet_plan7aNorm(TransCondTemporalUnet_plan7aN): # s作为qk，s作v替换s0
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.reduce = nn.Sequential(
            nn.Linear(Configs.dmodel, Configs.observation_dim),
            nn.LayerNorm(Configs.observation_dim),
        )
        self.reduce2 = nn.Sequential(
            nn.Linear(Configs.observation_dim*2, Configs.observation_dim),
            nn.LayerNorm(Configs.observation_dim),
        )
    





class TransCondTemporalUnet_plan7dN(TransCondTemporalUnet_plan6T1S): # s作为qk，s作v替换s0
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.reduce = nn.Linear(Configs.dmodel, Configs.observation_dim)
        self.reduce2 = nn.Linear(Configs.observation_dim*2, Configs.observation_dim)

    def forward(self, x, cond, time, bert, mask, history_obs, history_act):
        '''
            x : [ batch x horizon x transition ]
        '''
        # history_embd_cond = history_embd[:,-Configs.horizon, :] 
        history_obs_emb = self.state_embedding(history_obs)
        history_act_emb = self.action_embedding(history_act)
        history_embd = bert( history_obs_emb ,history_act_emb,  mask)  
        history_embd_cond = history_embd[:,-Configs.horizon, :] #
        history_embd_cond = self.reduce(history_embd_cond)


        # 重构x为embedding
        states = x[:, :, Configs.action_dim:]
        actions = x[:, :, :Configs.action_dim]

        states_embd = self.state_embedding(states)
        actions_embd = self.action_embedding(actions)

        # plan6作为对照实验，不添加历史信息,plan6a 和 6b需要将历史信息添加进去。
        x = torch.cat([ actions_embd, states_embd   ], -1)  

        comb = torch.cat([x[:,0, Configs.action_dim :],  history_embd_cond ], -1)

        cond = self.reduce2(comb)   # 将condition融合进去
        x[:, 0, Configs.action_dim:] = cond.clone()

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        # t = (t+history_aggregated)/2
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')[:,:,:Configs.transition_dim]
        return x

class TransCondTemporalUnet_plan7dS(TransCondTemporalUnet_plan7dN): # s作为qk，s作v替换s0
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.reduce = nn.Sequential(
            nn.Linear(Configs.dmodel, Configs.observation_dim),
            nn.Sigmoid(),
        )
        self.reduce2 = nn.Sequential(
            nn.Linear(Configs.observation_dim*2, Configs.observation_dim),
            nn.Sigmoid(),
        )
    
class TransCondTemporalUnet_plan7dM(TransCondTemporalUnet_plan7dN): # s作为qk，s作v替换s0
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.reduce = nn.Sequential(
            nn.Linear(Configs.dmodel, Configs.observation_dim),
            nn.Mish(),
        )
        self.reduce2 = nn.Sequential(
            nn.Linear(Configs.observation_dim*2, Configs.observation_dim),
            nn.Mish(),
        )
    

class TransCondTemporalUnet_plan7dNorm(TransCondTemporalUnet_plan7dN): # s作为qk，s作v替换s0
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.reduce = nn.Sequential(
            nn.Linear(Configs.dmodel, Configs.observation_dim),
            nn.LayerNorm(Configs.observation_dim),
        )
        self.reduce2 = nn.Sequential(
            nn.Linear(Configs.observation_dim*2, Configs.observation_dim),
            nn.LayerNorm(Configs.observation_dim),
        )
    



class TransCondTemporalUnet_plan8a(TransCondTemporalUnet_plan6T1S): # s作为qk，s作v替换s0
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.reduce = None
        self.reduce2 = None
    

    def forward(self, x, cond, time, bert, mask, history_obs, history_act):
        '''
            x : [ batch x horizon x transition ]
        '''
        # history_embd_cond = history_embd[:,-Configs.horizon, :] 
        history_obs_emb = self.state_embedding(history_obs)
        history_act_emb = self.action_embedding(history_act)
        history_embd = bert( history_obs_emb ,history_obs_emb,  mask)  
        history_embd_cond = history_embd[:,-Configs.horizon, :] #
        # history_embd_cond = self.reduce(history_embd_cond)


        # 重构x为embedding
        states = x[:, :, Configs.action_dim:]
        actions = x[:, :, :Configs.action_dim]

        states_embd = self.state_embedding(states)
        actions_embd = self.action_embedding(actions)

        # plan6作为对照实验，不添加历史信息,plan6a 和 6b需要将历史信息添加进去。
        x = torch.cat([ actions_embd, states_embd   ], -1)  

        # comb = torch.cat([x[:,0, Configs.action_dim :],  history_embd_cond ], -1)

        # cond = self.reduce2(comb)   # 将condition融合进去

        x[:, 0, Configs.action_dim:] = history_embd_cond.clone()

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        # t = (t+history_aggregated)/2
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')[:,:,:Configs.transition_dim]
        return x
    

class TransCondTemporalUnet_plan8b(TransCondTemporalUnet_plan6T1S): # s作为qk，s作v替换s0
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

        self.reduce = None
        self.reduce2 = nn.Sequential(
            nn.Linear(Configs.observation_dim*2, Configs.observation_dim),
            nn.Sigmoid(),
        )
    

    def forward(self, x, cond, time, bert, mask, history_obs, history_act):
        '''
            x : [ batch x horizon x transition ]
        '''
        # history_embd_cond = history_embd[:,-Configs.horizon, :] 
        history_obs_emb = self.state_embedding(history_obs)
        history_act_emb = self.action_embedding(history_act)
        history_embd = bert( history_obs_emb ,history_obs_emb,  mask)  
        history_embd_cond = history_embd[:,-Configs.horizon, :] #
        # history_embd_cond = self.reduce(history_embd_cond) # 此处出来后应该就是原始维度



        # 重构x为embedding
        states = x[:, :, Configs.action_dim:]
        actions = x[:, :, :Configs.action_dim]

        states_embd = self.state_embedding(states)
        actions_embd = self.action_embedding(actions)

        # plan6作为对照实验，不添加历史信息,plan6a 和 6b需要将历史信息添加进去。
        x = torch.cat([ actions_embd, states_embd   ], -1)  

        comb = torch.cat([x[:,0, Configs.action_dim :],  history_embd_cond ], -1)

        cond = self.reduce2(comb)   # 将condition融合进去
        
        x[:, 0, Configs.action_dim:] = cond.clone()

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        # t = (t+history_aggregated)/2
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')[:,:,:Configs.transition_dim]
        return x
    

class TransCondTemporalUnet_plan9(TemporalUnet):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

class TransCondTemporalUnet_plan9a(TransCondTemporalUnet_plan9):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)


class TransCondTemporalUnet_plan9b(TransCondTemporalUnet_plan9):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)



class TransCondTemporalUnet_plan9c(TransCondTemporalUnet_plan9):
    def __init__(self, horizon, transition_dim, cond_dim, dim=32, dim_mults=(1, 2, 4, 8), attention=False):
        super().__init__(horizon, transition_dim, cond_dim, dim, dim_mults, attention)

TransCondTemporalUnet_plan14bf = \
TransCondTemporalUnet_plan14g = \
TransCondTemporalUnet_plan14f2 = \
TransCondTemporalUnet_plan14f1 = \
TransCondTemporalUnet_plan14e = \
TransCondTemporalUnet_plan14d3 = \
TransCondTemporalUnet_plan14d2 = \
TransCondTemporalUnet_plan14d1 = \
TransCondTemporalUnet_plan15a = \
TransCondTemporalUnet_plan14b = \
TransCondTemporalUnet_plan14a2 = \
TransCondTemporalUnet_plan14a1 = \
TransCondTemporalUnet_plan12 = \
TransCondTemporalUnet_plan11f = \
TransCondTemporalUnet_plan11 = \
TransCondTemporalUnet_plan10d = \
TransCondTemporalUnet_plan10b = \
TransCondTemporalUnet_plan10a = \
TransCondTemporalUnet_plan10 = \
TransCondTemporalUnet_plan9dfn = \
TransCondTemporalUnet_plan9df = \
TransCondTemporalUnet_plan9cfn = \
TransCondTemporalUnet_plan9cf = \
TransCondTemporalUnet_plan9bfn = \
TransCondTemporalUnet_plan9bf = \
TransCondTemporalUnet_plan9afn = \
TransCondTemporalUnet_plan9af = \
TransCondTemporalUnet_plan9fn = \
TransCondTemporalUnet_plan9f = \
TransCondTemporalUnet_plan9d = \
TransCondTemporalUnet_plan9