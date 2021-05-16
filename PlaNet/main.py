import torch.cuda

from trainer import Trainer, get_env
from models import PlaNet
import wandb
from memory import ExperienceReplay
from tqdm import tqdm
import argparse
from utils import init_databuffer


config = {

'id':  'tests',
'seed': 1,
'disable cuda': False,
'env name':  'cheetah-run',
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


parser = argparse.ArgumentParser(description='Optional description of cuda device')
parser.add_argument('--gpu',type=int, choices=range(torch.cuda.device_count()))
parser.add_argument('--id', type=str)

arg = parser.parse_args()




if arg.gpu is not None:
    config['device'] = f'cuda:{arg.gpu}'

if arg.id is not None:
    config['id'] = arg.id

e = get_env(config)
model = PlaNet(config, env=e)
dataset = ExperienceReplay(config['experience size'], False, e.observation_size, e.action_size, config['bit depth'], config['device'])
init_databuffer(dataset, e, config['seed episodes'])

wandb.init(
    project='Pine',
    name=config['id'],
    config=config)



gym_teacher = Trainer(config, model, e, wandb_logger=wandb, buffer=dataset)
gym_teacher.train(1000)

