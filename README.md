# Planning with CDiffusion



The main branch contains code for training diffusion models and planning via value-function guided sampling on the D4RL locomotion environments.


## Prepare Environment

```
conda env create -f cdiff.yml
conda activate cdiffuser
```

## Using pretrained models

### Download pretrained models

<!-- Download pretrained diffusion models and value functions with:
```
./scripts/download_pretrained.sh
``` -->
Pretrained models are avilable from [this link](https://www.dropbox.com/scl/fo/nusf3zlwu1pxlkc1arn31/h?rlkey=oogb4t88jh9hicespg3385ttb&dl=0). Please download `logs` folder and move it to `./scripts/`, download `valuebase` folder and move it to anywhere you want.

Then setup the following environment variables:
```
export PYTHONPATH=PATH_TO_CDiffuser
export VALBASE=PATH_TO_valuebase
```



### Planning

To plan with guided sampling, run:

```
python  plan_guided.py --dataset hopper-medium-v2 --nums_eval 1 --branch plan2_hard --upperbound 0.65 --lowerbound 0.2 --slope 800 --contrastweigth 0.001 --seed 1000 --device cuda:3
```

## Training from scratch

1. Train a diffusion model with:
```
python train.py --dataset halfcheetah-medium-expert-v2
```

The default hyperparameters are listed in `config/locomotion.py` and `scripts/base.py`.
You can override any of them with flags, eg, `--n_diffusion_steps 100`.

2. Train a value function with:
```
python train_values.py --dataset halfcheetah-medium-expert-v2
```
See [locomotion:values](main/config/locomotion.py#L67-L108) for the corresponding default hyperparameters.


3. Plan using your newly-trained models with the same command as in the pretrained planning section, simply replacing the logbase to point to your new models:
```
python plan_guided.py --dataset halfcheetah-medium-expert-v2 --logbase logs
```
See [locomotion:plans](main/config/locomotion.py#L110-L149) for the corresponding default hyperparameters.

