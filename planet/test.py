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
    'batch size' : 50,
    'belief size' : 200,
    'state size'  : 30,
    'embedding size' : 1024,
    'action size' : 1,

    'activation function': 'relu',
    'action repeat': 2,
    'collect interval' : 1000,
    'minimum standard devation' : 0.1,
    'hidden size' : 200,
    'candidates' : 1000,
    'top candidates' : 100,
    'planning horizon' : 12,
    'optimisation iterations' : 10,
    'max episode length' : 1000,
    'bit depth' : 5,
    'action noise' : 0.3,
    'episode size' : 1000000,
    'init episodes' : 5

}

planet = main.Planet(config)

trainer = pl.Trainer(
    gpus=0,
    max_epochs=1000,
    val_check_interval=100
)

trainer.fit(planet)