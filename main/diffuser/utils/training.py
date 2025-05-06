import os
import copy
import numpy as np
import torch
import einops
import pdb
import wandb
from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs
# from diffuser import mingpt
from diffuser.models import  *
from datetime import datetime,timedelta
from config.locomotion_config import Configs
from myutils import convert_log

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=1000,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        ie_dataset=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        # self.args = args

        self.device = list(self.model.model.parameters())[0].device


        # self.inverse_trans_model = TransionInverseModel(Configs.embd_for_ie, Configs.observation_dim, Configs.action_dim).to(Configs.device)
        # self.inverse_trans_model_ema = copy.deepcopy(self.inverse_trans_model)


        # self.ie_dataset = ie_dataset if ie_dataset else dataset
        # self.ie_dataloader = cycle(torch.utils.data.DataLoader(
        #     self.ie_dataset, batch_size=train_batch_size, num_workers=0, shuffle=True, pin_memory=True
        # ))


        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=20, shuffle=True, pin_memory=True, prefetch_factor=20
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        # self.logdir = results_folder
        self.bucket = results_folder
        self.n_reference = n_reference

        self.reset_parameters()
        self.epoch = 0
        self.step = 0
        # self.logfile =  os.path.join(self.bucket, f"training_log.log")
        import socket
        self.log(f"Task activated at {socket.gethostname()}.")

    def log(self, info, flush = True):
        Configs.logger.log(info, flush)
        return
    
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    # def buld_str(self, info_pack):
    #     return str.join(  f"{key}: {info_pack[key]:8.4f}" for key in info_pack.keys()  )

    def train(self, n_train_steps):
        device = Configs.device
        assert device
        timer = Timer()

        self.log(f"======================  Start training  ======================")

        if Configs.recover:
            load_latest = os.path.join(self.bucket, "state_latest.pt")
            if os.path.exists(load_latest):
                self.load("latest")
            else:
                self.log(f"Failed to load {load_latest}.")
        else:
            self.log("!!!!!   Not loading. !!!!!")

        self.model.train()
        while True:
            dataset_iter = iter(self.dataloader)
            self.log("Rebuilt dataset iter.")
            while True:
                try:
                    Configs.step = self.step
                    for i in range(self.gradient_accumulate_every):
                        batch = next(dataset_iter) 
                        # traj  cond posi_s nega_s posi_v nega_v his
                        batch = batch_to_device(batch, device)
                        loss,  infos = self.model.loss(*batch)
                        loss = loss / self.gradient_accumulate_every
                        loss.backward()

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.step % self.update_ema_every == 0:
                        self.step_ema()

                    if self.step % self.save_freq == 0:
                        label = self.step // self.label_freq * self.label_freq
                        self.save(label)
                        if self.step > 0:
                            convert_log(self.bucket)
                    if self.step % self.log_freq == 0:
                        time_cost = timer()
                        now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        self.log(f"======================  STEP {self.step}/{n_train_steps} Epoch {self.epoch} ======================")
                        self.log(f'loss:{loss:8.4f}')
                        self.log(" | ".join( [ f"{key}: {infos[key]:8.4f}" for key in infos.keys() ] ))
                        self.log(f"####TensorboardInfos####|{self.step}|" + "|".join( [ f"{key}@{infos[key]:.4f}" for key in infos.keys() ] ))
                        stage_2_left_seconds = max((n_train_steps  - self.step)//self.log_freq,0)*time_cost
                        eta_2 = str(timedelta(seconds=stage_2_left_seconds)) 
                        self.log(f"============= time:{time_cost:4.4f} ETA:{eta_2}  =============\n", flush=True)

                        if Configs.wandb:
                            wandb.log(infos)



                    self.step += 1
                    if self.step >= n_train_steps:
                        break
                except StopIteration:
                    self.log(f"Epoch {self.epoch} finished.")
                    break
            self.epoch += 1
            if self.step >= n_train_steps:
                break


    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'epoch': self.epoch
            # 'id_model': self.inverse_trans_model.state_dict(),
            # 'id_ema': self.inverse_trans_model_ema.state_dict()
        }
        savepath = os.path.join(self.bucket, f'state_{epoch}.pt')
        torch.save(data, savepath)

        savepath_latest = os.path.join(self.bucket, f'state_latest.pt')
        torch.save(data, savepath_latest)

        self.log(f'[ utils/training ] Saved model to {savepath}', flush=True)
        # if self.bucket is not None:
        #     sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        from .serialization import get_latest_epoch

        loadpath = os.path.join(self.bucket, f'state_{epoch}.pt')
        self.log(f"Loading from {loadpath}.")
        

        data = torch.load(loadpath, map_location=torch.device('cpu'))

        self.step = data['step']
        if 'epoch' in data.keys():
            self.epoch = data['epoch']
        # if isinstance(self.model, TransionInverseModel):
        #     self.inverse_trans_model.load_state_dict( data['id_model'] )
        #     self.inverse_trans_model_ema.load_state_dict( data['id_ema']  )
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

        self.log(f"Loaded epoch {self.step}.")

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.bucket, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self,device , batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, device)

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.bucket, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)
