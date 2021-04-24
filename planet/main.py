import pytorch_lightning as pl
import models
import data
import torch
import utils
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict



from dm_control import suite
from dm_control.suite.wrappers import pixels

class Planet(pl.LightningModule):

    def __init__(self, config):
        super(Planet, self).__init__()

        self.config = config
        self.buffer = data.EpisodeBuffer(config)
        self.broker = data.Broker('ball_in_cup-catch', self.buffer, config)


        self.tm = models.TransitionModel(config)
        self.om = models.ObservationModel(config)
        self.rm = models.RewardModel(config)
        self.encoder = models.Encoder(config)
        self.planner = models.Planner(config, self.tm, self.rm)

        self.init_dataset()

        self.collect_interval = config['collect interval']
        self.action_repeat = config['action repeat']
        self.max_episode_length = config['max episode length']

        self.current_step = 0

    def init_dataset(self):

        for s in range(1, self.config['init episodes'] + 1):
            self.broker.random_episode()




    def training_step(self, batch, n_batch):

        batch_size = batch[0][0].shape[0]
        observation, actions, rewards, nonterminals = batch
        observation = observation.squeeze(dim=0)
        actions = actions.squeeze(dim=0)
        rewards = rewards.squeeze(dim=0)
        nonterminals = nonterminals.squeeze(dim=0)
        init_belief, init_state = torch.zeros(batch_size, self.config['belief size'], device=self.device), torch.zeros(batch_size, self.config['state size'], device=self.device)

        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = \
            self.tm(init_state, actions[:-1], init_belief, utils.bottle(self.encoder, (observation[1:],)), nonterminals[:-1])

        observation_loss = F.mse_loss(utils.bottle(self.om, (beliefs, posterior_states)), observation[1:],
                                      reduction='none').sum((2, 3, 4)).mean(dim=(0, 1))

        predicted_rewards = utils.bottle(self.rm, (beliefs, posterior_states))
        reward_loss = F.mse_loss(predicted_rewards, rewards[:-1],
                                 reduction='none').mean(dim=(0, 1))

        loss = (observation_loss + reward_loss)

        if self.current_step == self.collect_interval:
            self.current_step = 0
            self.data_collection()

        else: self.current_step += 1

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss



    @torch.no_grad()
    def data_collection(self):

        observation, to_store = self.broker.reset()
        rewards = 0
        belief = torch.zeros(1, self.config['belief size'], device=self.device)
        posterior = torch.zeros(1, self.config['state size'], self.device)
        action = torch.zeros(1, self.config['action space'])

        for step in range(self.max_episode_length // self.action_repeat):
            belief, posterior, action, observation, reward, done, to_store = self.broker.play_step(belief,
                                                                                      posterior,
                                                                                      action,
                                                                                      observation,
                                                                                      store=to_store,
                                                                                      give_uint=True)
            rewards += reward

            if done:
                break




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        obs, post, belief, action = x
        encoded_belief = self.encoder(belief)
        belief, _, _, _, post, _, _ = self.tm(post, action.unsqueeze(dim=0), encoded_belief.unsqueeze(dim=0))

        belief, post = belief.unsqueeze(dim=0),post.unsqueeze(dim=0)

        action = self.planner(belief, post)


    def train_dataloader(self) -> DataLoader:
        dataset = data.Dataset(self.buffer)

        dataloader = DataLoader(dataset=dataset, batch_size=self.config['batch size'])
        return dataloader


    def configure_optimizers(
            self,
    ):
        paramlist = list(self.tm.parameters()) + list(self.om.parameters()) + list(self.rm.parameters()) + list(self.encoder.parameters())

        optimizer = torch.optim.Adam(paramlist, lr=0.001, eps=0.0001)

        return [optimizer]






