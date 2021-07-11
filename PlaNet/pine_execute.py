import torch.cuda

from trainer import Trainer, get_env, PineTrainer
from models import PlaNet, Pine
import wandb
from memory import ExperienceReplay
from tqdm import tqdm
import argparse
from utils import init_databuffer
import env


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

initial_envs = [
    ('cartpole-balance', 'Balance an unactuated pole by applying forces to a cart at its base, starting with the pole upwards and with non-sparse rewards', 'manual_models/cartpole_balance.pth'),
    ('finger-spin', 'A 3DoF planar finger is required to rotate a toy body on an unactuated hinge, the body must be continually rotated or spun', 'manual_models/finger_spin.pth'),
    ('walker-walk', 'A planar walker with two legs and a torso, recieve rewards for standing the torso upright and in a certain high, while also walking', 'manual_models/walker.pth'),
    ('reacher-easy', 'A simple two link planar reacher, get rewards when the end effector is inside the large target area', 'manual_models/reacher.pth')
]

test_envs = [

    ('cartpole-swingup', 'Starting with the pole downwards, swing the unactuated pole up and balance it by applying forces to the cart at the base, with non sparse-rewards'),
    ('walker-run', 'Make the planar walker, with two legs, run fast forward without falling to the ground, meaning the torso must be upright and at the correct height'),
    ('pendulum-swingup', 'Classic pendulum swingup task, swing the pendulum up and balance it, receive sparse rewards while the pendulum is within 30 degrees of the vertical'),
    ('finger-spin', 'Use the finger to spin the a toy around a hinge, keep spinning to get more rewards'),
    ('acrobot-swingup', 'The underactuated double pendulum, torque applied to the second joint, where the goal is to swing up and balance')

]


pine_dict = torch.load('manual_models/walker.pth', map_location=torch.device(config['device']))
obs, enc = pine_dict['planet_model'][2], pine_dict['planet_model'][3]

Pine_dict = {'networks' : [enc, obs], 'agents' : []}


pine = Pine(config, saved_dict=Pine_dict)


for env_name, desc, path in initial_envs:
    config['action repeat'] = env.CONTROL_SUITE_ACTION_REPEATS['env_name'] if env_name in env.CONTROL_SUITE_ACTION_REPEATS else 4
    environment = get_env(config, env_name)

    dicts = torch.load(path, map_location=torch.device(config['device']))
    dicts = (dicts['planet_model'][1], dicts['planet_model'][0])

    pine.manual_agent_add(dicts, desc, environment.reset(), action_size=environment.action_size, name=env_name)



trainer = PineTrainer(config, pine)

trainer.train(test_envs, 200)





