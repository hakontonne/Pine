import pytorch_lightning as pl
import models
import data
import torch
import utils
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence


from pytorch_lightning.loggers import WandbLogger




from dm_control import suite
from dm_control.suite.wrappers import pixels

class Planet(pl.LightningModule):

    def __init__(self, config):
        super(Planet, self).__init__()

        self.config = config
        self.batch_size = config['batch size']
        self.buffer = data.EpisodeBuffer(config)
        self.broker = data.Broker('cartpole-balance', self.buffer, config)


        self.tm = models.TransitionModel(config)
        self.om = models.ObservationModel(config)
        self.rm = models.RewardModel(config)
        self.encoder = models.Encoder(config)
        self.planner = models.Planner(config, self.tm, self.rm)

        self.init_dataset()
        self.broker._init_networks(self.planner, self.tm, self.encoder)

        self.collect_interval = config['collect interval']
        self.action_repeat = config['action repeat']
        self.max_episode_length = config['max episode length']

        self.current_step = 0
        self.free_nats = torch.full((1, ), 3, dtype=torch.float32, device=self.device)

    def init_dataset(self):

        for s in range(1, self.config['init episodes'] + 1):
            self.broker.random_episode()




    def training_step(self, batch, n_batch):


        observation, actions, rewards, nonterminals = batch
        observation = observation.squeeze(dim=0)
        actions = actions.squeeze(dim=0)
        rewards = rewards.squeeze(dim=0)
        nonterminals = nonterminals.squeeze(dim=0)
        init_belief, init_state = torch.zeros(self.batch_size, self.config['belief size'], device=self.device),\
                                  torch.zeros(self.batch_size, self.config['state size'], device=self.device)

        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = \
            self.tm(init_state, actions[:-1], init_belief, utils.bottle(self.encoder, (observation[1:],)), nonterminals[:-1])

        observation_loss = F.mse_loss(utils.bottle(self.om, (beliefs, posterior_states)), observation[1:],
                                      reduction='none').sum((2, 3, 4)).mean(dim=(0, 1))

        predicted_rewards = utils.bottle(self.rm, (beliefs, posterior_states))
        reward_loss = F.mse_loss(predicted_rewards, rewards[:-1],
                                 reduction='none').mean(dim=(0, 1))
        free_nats = torch.full((1,), 3, dtype=torch.float32, device=self.device)
        kl_loss = torch.max(
            kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2),
            free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out

        self.log('observation loss', observation_loss, on_step=True, on_epoch=True, logger=True)
        self.log('reward loss', reward_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('kl loss', kl_loss, on_step=True, on_epoch=True, logger=True)



        loss = (observation_loss + reward_loss + kl_loss)

        if self.current_step == self.collect_interval:
            self.current_step = 0
            self.data_collection()
            self.validation_step(0, 0)

        self.current_step += 1

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss



    def validation_step(self, batch, batch_idx):

        total_reward = 0
        for _ in range(2):

            observation, __ = self.broker.reset()
            observation = observation.to(self.device)

            belief = torch.zeros(1, self.config['belief size'], device=self.device)
            posterior = torch.zeros(1, self.config['state size'], device=self.device )
            action = torch.zeros(1, self.config['action space'], device=self.device)
            for step in range(self.max_episode_length // self.action_repeat):

                belief, posterior, action, observation, reward, done = self.broker.play_step(belief, posterior,
                                                                                                       action,
                                                                                                       observation)
                observation = observation.to(self.device)
                total_reward += reward

                if done:
                    break

        self.log('val_reward', total_reward, on_step=True, on_epoch=True, prog_bar=True, logger=True)


    @torch.no_grad()
    def data_collection(self):

        observation, to_store = self.broker.reset()
        rewards = 0
        print('Call to data_collection now')
        observation = observation.to(self.device)
        belief = torch.zeros(1, self.config['belief size'], device=self.device)
        posterior = torch.zeros(1, self.config['state size'], device=self.device)
        action = torch.zeros(1, self.config['action space'], device=self.device)

        for step in range(self.max_episode_length // self.action_repeat):


            belief, posterior, action, observation, reward, done, to_store = self.broker.play_step(belief,
                                                                                      posterior,
                                                                                      action,
                                                                                      observation, explore=True,
                                                                                      store=to_store, give_uint=True)
            observation = observation.to(self.device)
            rewards += reward

            if done:
                break

        self.log('train reward', rewards, on_epoch=True, prog_bar=True)




    def forward(self, x: torch.Tensor) -> torch.Tensor:
        obs, post, belief, action = x
        encoded_belief = self.encoder(belief)
        belief, _, _, _, post, _, _ = self.tm(post, action.unsqueeze(dim=0), encoded_belief.unsqueeze(dim=0))

        belief, post = belief.unsqueeze(dim=0),post.unsqueeze(dim=0)

        action = self.planner(belief, post)

        return action


    def train_dataloader(self) -> DataLoader:
        dataset = data.Dataset(self.buffer, self.batch_size)

        dataloader = DataLoader(dataset=dataset, batch_size=self.config['batch size'], num_workers=8, pin_memory=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataset = data.Dataset(self.buffer, self.batch_size)
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size)

        return dataloader

    def configure_optimizers(
            self,
    ):
        paramlist = list(self.tm.parameters()) + list(self.om.parameters()) + list(self.rm.parameters()) + list(self.encoder.parameters())

        optimizer = torch.optim.Adam(paramlist, lr=0.001, eps=0.0001)

        return [optimizer]


    def test_agent(self, broker):

        images = []
        observation, to_store = broker.reset()
        rewards = 0
        observation = observation.to(self.device)

        belief = torch.zeros(1, self.config['belief size'], device=self.device)
        posterior = torch.zeros(1, self.config['state size'], device=self.device)
        action = torch.zeros(1, self.config['action space'], device=self.device)
        c = 0
        max_iters = 1000
        done = False

        while not done and (c < max_iters):
            belief, posterior, action, observation, reward, done, to_store = broker.play_step(belief,
                                                                                                   posterior,
                                                                                                   action,
                                                                                                   observation,
                                                                                                   give_uint=True)
            observation = observation.to(self.device)
            rewards += reward

            images.append(to_store)

        return images, rewards









