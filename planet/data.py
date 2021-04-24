import pytorch_lightning as pl
import collections
from recordclass import recordclass, RecordClass
import numpy as np
import torch
from torch.utils.data import IterableDataset
from dm_control import suite
from dm_control.suite.wrappers import pixels
import itertools


import utils
class DeepmindDatamodule(pl.LightningDataModule):

    def __init__(self, config):
        self.observation_size   = config['observation size']
        self.action_size        = config['action size']
        self.bitdepth           = config['bitdepth']



class Episode(RecordClass):
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor

class EpisodeBuffer:


    def __init__(self, config):
        self.bitdepth = config['bit depth']
        self.buffer = collections.deque(maxlen=config['episode size'])


    def __len__(self):
        return len(self.buffer)

    def append(self, e: Episode) -> None:


        e.done = not e.done


        self.buffer.append(e)


    def sample(self, batch_size: int):

        observations, actions, rewards, nonterminals = [], [], [], []
        idxs = np.random.choice(len(self.buffer)-batch_size, batch_size, replace=False)
        #indicies = [np.arange(i, len(self.buffer)-batch_size) for i in idxs]
        for idx in idxs:
            observation, action, reward, nonterminal = zip(*list(itertools.islice(self.buffer, idx, idx+batch_size)))
            observation = utils.image_to_tensor(list(observation))
            observations.append(observation)
            actions.append(torch.stack(action))
            rewards.append(torch.Tensor(reward))
            nonterminals.append(torch.Tensor(nonterminal).unsqueeze(dim=1))



        return torch.stack(observations), torch.stack(actions), torch.stack(rewards), torch.stack(nonterminals)


class Dataset(IterableDataset):

    def __init__(self, buffer: EpisodeBuffer, sample_size: int = 50):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        #observation, action, reward, nontermials = self.buffer.sample(self.sample_size)
        #for i in range(len(nontermials)):
        #    yield observation[i], action[i], reward[i], nontermials[i]
        yield self.buffer.sample(self.sample_size)

class Broker:
    def __init__(self, gym_name, replay_buffer, planner, transition_model, encoder, config):
        self._action_repeat = config['action repeat']
        self._max_episode_length = config['max episode length']
        self._bit_depth = config['bit depth']
        self._action_noise = config['action noise']

        self._init_networks(planner, transition_model, encoder)
        self.env = self._init_env(gym_name)
        self.buffer = replay_buffer
        self.reset()

        config['action space'] = self.action_space

    def __init__(self,  gym_name, replay_buffer, config):
        self._action_repeat = config['action repeat']
        self._max_episode_length = config['max episode length']
        self._bit_depth = config['bit depth']
        self._action_noise = config['action noise']

        self.env = self._init_env(gym_name)
        self.buffer = replay_buffer
        self.reset()

        config['action space'] = self.action_space

    def _init_networks(self, planner, transition_model, encoder):
        self.planner = planner
        self.transition_model = transition_model
        self.encoder = encoder


    def _init_env(self, gym_name):
        domain, task = gym_name.split('-')

        env = suite.load(domain_name=domain, task_name=task)
        env = pixels.Wrapper(env)
        self.action_space = env.action_spec().shape[0]

        return env

    def reset(self, storage=False):
        self.t = 0
        self.state = self.env.reset()
        img = self.env.physics.render(camera_id=0)
        return utils.image_to_tensor(img), img
    def close(self):
        self.env.close()


    @torch.no_grad()
    def play_step(self, belief, posterior_state, action, observation, explore=False, store=None, give_uint=False):
        belief, _, _, _, posterior_state, _, _ = self.transition_model(posterior_state, action.unsqueeze(dim=0), belief,
                                                self.encoder(observation).unsqueeze(dim=0))
        belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)
        action = self.planner(belief, posterior_state)
        if explore:
            action = action + self._action_noise * torch.randn_like(action)
        next_obs, reward, done = self.step(action[0].cpu())

        action.clamp_(self.env.action_space.low[0], self.env.action_space.high[0])


        if store: self.buffer.append(Episode(store, action, reward, done))

        if give_uint:
            return belief, posterior_state, action, utils.image_to_tensor(next_obs), reward, done, next_obs
        return belief, posterior_state, action, utils.image_to_tensor(next_obs), reward, done


    @torch.no_grad()
    def step(self, action):
        action = action.detach().numpy()
        r = 0

        for k in range(self._action_repeat):
            self.state = self.env.step(action)
            r += self.state.reward
            self.t += 1
            done = self.state.last() or self.t == self._max_episode_length
            if done:
                self.reset()
                break



        return self.env.physics.render(camera_id=0), r, done

    def random_episode(self):

        done = False
        _, obs = self.reset()
        t = 0

        while not done:
            action = utils._random_action(self.env.action_spec())
            next_obs, reward, done = self.step(action)
            exp = Episode(obs, action, reward, done)
            obs = next_obs
            self.buffer.append(exp)
            t += 1



