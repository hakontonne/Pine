import data
import numpy as np
import main
import pytorch_lightning as pl

"""
cfg = {
    'bitdepth' : 1,
    'episode size' : 1500,
    'gym name' = 'walker-walk'
}
"""

from pytorch_lightning.loggers import WandbLogger



gpus = [4,5]


#
# for i in range(1500):
#     rand = data.Episode(np.random.random(50), np.random.random(50), np.random.random(50), np.random.random(50))
#     buff.append(rand)
#
#
# obs, act, rew, don = dataset.__iter__()
# print(buff)
# print(dataset)




config = {
    'batch size' :                  64,
    'chunk size' :                  50,
    'belief size' :                 200,
    'state size'  :                 30,
    'embedding size' :              1024,


    'activation function':          'relu',
    'action repeat':                4,
    'collect interval' :            100,
    'minimum standard devation' :   0.1,
    'hidden size' :                 200,
    'candidates' :                  1000,
    'top candidates' :              100,
    'planning horizon' :            12,
    'optimisation iterations' :     10,
    'max episode length' :          1000,
    'bit depth' :                   5,
    'action noise' :                0.3,
    'episode size' :                1000000,
    'init episodes' :               20,
    'num devices' :                 0

}

wandb_logger = WandbLogger(name='cartpole_balance1', project='Pine')
planet = main.Planet(config)

trainer = pl.Trainer(
    gpus=gpus, accelerator='ddp',
    max_steps=1000,
    val_check_interval=50,
    logger=wandb_logger
)

trainer.fit(planet)

br = planet.broker
imgs, reward = planet.test_agent(br)

import cv2
fourcc = cv2.VideoWriter_fourcc(*'XVID')
vid = cv2.VideoWriter('cheta3.avi', fourcc, 20.0, (320, 240))

for img in imgs:
    vid.write(img)

vid.release()



