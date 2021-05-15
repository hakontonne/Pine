import torch.cuda

from trainer import Trainer, get_env, PineTrainer
from models import PlaNet, Pine
import wandb
from memory import ExperienceReplay
from tqdm import tqdm
import argparse
from utils import init_databuffer


parser = argparse.ArgumentParser(description='Optional description of cuda device')
parser.add_argument('--gpu',type=int, choices=range(torch.cuda.device_count()))
parser.add_argument('--id', type=str)

arg = parser.parse_args()

config = {

'id':  'pine_test',
'seed': 1,
'disable cuda': False,
'env name':  None,
'action repeat': 8,
'symbolic env': False,
'max episode length': 1000,
'experience size': 1000000,
'embedding size': 1024,
'hidden size': 200,
'belief size': 200,
'state size': 30,

'action noise': 0.3,
'episodes': 1000,
'seed episodes': 5,
'collect interval': 100,
'batch size': 50,
'chunk size': 50,
'overshooting distance': 50,
'overshooting kl beta': 0,
'overshooting reward scale': 0,
'global kl beta': 0,
'free nats': 3,
'bit depth': 5,
'learning rate': 0.001,
'learning rate schedule': 0,
'adam epsilon': 0.0001,
'grad clip norm': 1000,
'planning horizon': 12,
'optimisation iters': 10,
'candidates': 1000,
'top candidates': 100,
'test': False,
'test interval': 25,
'test episodes': 10,
'checkpoint interval': 50,
'checkpoint experience': False,

'render': False,
'device': 'cuda:0',

}

if arg.gpu is not None:
    config['device'] = f'cuda:{arg.gpu}'

if arg.id is not None:
    config['id'] = arg.id



envs = ['cartpole-balance', 'cartpole-balance_sparse' ,  'cartpole-swingup', 'cartpole-swingup_sparse', 'reacher-easy', 'reacher-hard', 'walker-walk', 'walker-stand']

wandb.init(
    project='Pine',
    name='Pine_test1',
    config=config)


pine = Pine(config)

trainer = PineTrainer(config, pine, wandb_logger=wandb)

trainer.train(envs, 100)
