import argparse
from math import inf
import os
import numpy as np
import torch
import wandb
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher
from memory import ExperienceReplay
from models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel
from planner import MPCPlanner
from utils import lineplot, write_video, init_databuffer

def get_env(config, env_name=None):
    return Env(config['env name'] if env_name is None else env_name, config['symbolic env'], config['seed'], config['max episode length'], config['action repeat'], config['bit depth'])


class BertTrainer():

    def __init__(self, model):
        self.model = model
        self.loss_function = torch.nn.BCEWithLogitsLoss()
        self.optim = optim.Adam(model.parameters(), lr=0.001, eps=0.0001)

    def train(self, dataset, n_epochs):
        losses = []
        for i in tqdm(range(n_epochs)):
            epoch_loss = 0
            for x, y in tqdm(dataset, desc="In dataset"):

                loss = self.forward_step(x.to('cuda:5'), y.to('cuda:5'))
                self.optim.zero_grad()
                loss.backward(retain_graph=True)
                self.optim.step()

                epoch_loss += loss.detach().cpu().numpy()

            losses.append(epoch_loss)

        return losses






    def forward_step(self, x, y):
        pred = self.model(x)
        return self.loss_function(pred, y)





class Trainer():

    def __init__(self, config, model, env, wandb_logger=None, buffer=None):

        self.env = env
        self.planet_model = model

        self.dataset = buffer
        self.logger = wandb_logger
        self.optimiser = model.optimiser

        self.env_name = env.env_name
        self.collection_interval = config['collect interval']
        self.grad_clip_norm = config['grad clip norm']
        self.batch_size = config['batch size']
        self.chunk_size = config['chunk size']
        self.max_episode_length = config['max episode length']
        self.action_repeat = config['action repeat']
        self.test_episodes = config['test episodes']
        self.test_interval = config['test interval']
        self.checkpoint_interval = config['checkpoint interval']
        self.seed = config['seed']
        self.id = config['id']

    def train(self, iters):

        if not os.path.exists(f'videos/{self.id}'):
            os.makedirs(f'videos/{self.id}')

        if not os.path.exists(f'checkpoints/{self.id}'):
            os.makedirs(f'checkpoints/{self.id}')


        for i in tqdm(range(iters), desc='Running training'):
            acc_obs_loss = 0
            acc_reward_loss = 0
            acc_kl_loss    = 0
            for j in range(self.collection_interval):
                observations, actions, rewards, nonterminals = self.dataset.sample(self.batch_size, self.chunk_size)

                pred_observation, pred_reward, kl_loss = self.planet_model.forward(observations, actions, nonterminals)

                observation_loss = F.mse_loss(pred_observation, observations[1:], reduction='none').sum(dim=(2, 3, 4)).mean(dim=(0, 1))
                reward_loss = F.mse_loss(pred_reward, rewards[:-1], reduction='none').mean(dim=(0, 1))
                self.optimiser.zero_grad()
                (observation_loss + reward_loss + kl_loss).backward()
                nn.utils.clip_grad_norm_(self.planet_model.param_list, self.grad_clip_norm, norm_type=2)
                self.optimiser.step()

                acc_kl_loss += kl_loss.item()
                acc_obs_loss += observation_loss.item()
                acc_reward_loss += reward_loss.item()


            self.logger.log({'train_obs_loss' : acc_obs_loss, 'train_reward_loss': acc_reward_loss}, step=i)

            with torch.no_grad():
                observation, total_reward = self.env.reset(), 0
                belief, posterior_state, action = torch.zeros(1, self.planet_model.belief_size, device=self.planet_model.device),\
                                                  torch.zeros(1, self.planet_model.state_size,  device=self.planet_model.device),\
                                                  torch.zeros(1, self.env.action_size, device=self.planet_model.device)

                for _ in range(self.max_episode_length//self.action_repeat):

                    belief, posterior_state, action = self.planet_model.get_action(belief, posterior_state, action, observation.to(device=self.planet_model.device), explore=True)

                    next_obs, reward, done = self.env.step(action[0].cpu())
                    self.dataset.append(observation, action.cpu(), reward, done)
                    total_reward += reward
                    observation = next_obs

                self.logger.log({'train_reward': total_reward}, step=i)


            if i % self.test_interval == 0:
                self.planet_model.eval()

                test_envs = EnvBatcher(env=(self.env, self.test_episodes))


                with torch.no_grad():
                    observation, total_rewards, video_frames = test_envs.reset(), np.zeros((self.test_episodes,)), []
                    belief, posterior_state, action = torch.zeros(self.test_episodes, self.planet_model.belief_size,device=self.planet_model.device),\
                                                      torch.zeros(self.test_episodes, self.planet_model.state_size, device=self.planet_model.device),\
                                                      torch.zeros(self.test_episodes, self.env.action_size,         device=self.planet_model.device)

                    for t in range(self.max_episode_length // self.action_repeat):
                        belief, posterior_state, action = self.planet_model.get_action(belief, posterior_state, action, observation.to(device=self.planet_model.device), action_max=self.env.action_range[1], action_min=self.env.action_range[0])
                        next_obs, reward, done = test_envs.step(action.cpu())

                        total_rewards += reward.numpy().sum()
                        video_frames.append(make_grid(torch.cat([observation, self.planet_model.observation_model(belief, posterior_state).cpu()], dim=3) + 0.5, nrow=5).numpy())

                        observation = next_obs
                        if done.sum().item() == self.test_episodes:
                            break



                    write_video(video_frames, f'test_episode_{i}', './videos/'+self.id)
                    self.logger.log( {'test_reward' : total_rewards.sum()}, step=i)

                    self.logger.log({'test_episode_video': wandb.Video(f'videos/{self.id}/test_episode_{i}.mp4', format='gif')}, step=i)

                self.planet_model.train()
                test_envs.close()

            if i % self.checkpoint_interval == 0:
                torch.save({'planet_model' : self.planet_model.state_dict()}, os.path.join(f'checkpoints/{self.id}', 'models_%d.pth' % i))



    def iter_batched_test(self, iters):
        rewards = 0
        for _ in iters:
            rewards += self.batched_test()

        return rewards / iters

    def batched_test(self):

        self.planet_model.eval()
        env_args = (self.env_name, False, self.seed, self.env.max_episode_length, self.env.action_repeat, self.env.bit_depth)
        test_envs = EnvBatcher(Env, env_args, {}, self.test_episodes)

        with torch.no_grad():
            observation, total_rewards, video_frames = test_envs.reset(), np.zeros((self.test_episodes,)), []
            belief, posterior_state, action = torch.zeros(self.test_episodes, self.planet_model.belief_size,
                                                          device=self.planet_model.device), \
                                              torch.zeros(self.test_episodes, self.planet_model.state_size,
                                                          device=self.planet_model.device), \
                                              torch.zeros(self.test_episodes, self.env.action_size,
                                                          device=self.planet_model.device)
            for t in range(self.max_episode_length // self.action_repeat):
                belief, posterior_state, action = self.planet_model.get_action(belief, posterior_state, action,
                                                                               observation.to(
                                                                                   device=self.planet_model.device),
                                                                               action_max=self.env.action_range[1],
                                                                               action_min=self.env.action_range[0])
                next_obs, reward, done = test_envs.step(action.cpu())
                total_rewards += reward.numpy().sum()

                observation = next_obs
                if done.sum().item() == self.test_episodes:
                    break

        self.planet_model.train()
        test_envs.close()

        return total_rewards



class PineTrainer():

    def __init__(self, config, model):

        self.model = model



        self.collection_interval = config['collect interval']
        self.grad_clip_norm = config['grad clip norm']
        self.batch_size = config['batch size']
        self.chunk_size = config['chunk size']
        self.max_episode_length = config['max episode length']

        self.test_episodes = config['test episodes']
        self.test_interval = config['test interval']
        self.checkpoint_interval = config['checkpoint interval']
        self.seed = config['seed']
        self.id = config['id']
        self.config = config
        self.device = config['device']


    def train_new_agent(self, env, dataset, iters, logger):

        action_repeat = env.action_repeat
        for i in tqdm(range(iters), desc='Running training'):
            acc_obs_loss = 0
            acc_reward_loss = 0
            acc_kl_loss    = 0
            for j in range(self.collection_interval):
                observations, actions, rewards, nonterminals = dataset.sample(self.batch_size, self.chunk_size)

                pred_observation, pred_reward, kl_loss = self.model.planet.forward(observations, actions, nonterminals)

                observation_loss = F.mse_loss(pred_observation, observations[1:], reduction='none').sum(dim=(2, 3, 4)).mean(dim=(0, 1))
                reward_loss = F.mse_loss(pred_reward, rewards[:-1], reduction='none').mean(dim=(0, 1))


                self.model.planet.optimiser.zero_grad()
                (observation_loss + reward_loss + kl_loss).backward()
                nn.utils.clip_grad_norm_(self.model.planet.param_list, self.grad_clip_norm, norm_type=2)
                self.model.planet.optimiser.step()

                acc_kl_loss += kl_loss.item()
                acc_obs_loss += observation_loss.item()
                acc_reward_loss += reward_loss.item()


            logger.log({'train_obs_loss' : acc_obs_loss, 'train_reward_loss': acc_reward_loss}, step=i)

            with torch.no_grad():
                observation, total_reward = env.reset(), 0
                belief, posterior_state, action = torch.zeros(1, self.model.planet.belief_size, device=self.model.device),\
                                                  torch.zeros(1, self.model.planet.state_size,  device=self.model.device),\
                                                  torch.zeros(1, env.action_size, device=self.model.device)

                for _ in range(self.max_episode_length//env.action_repeat):

                    belief, posterior_state, action = self.model.planet.get_action(belief, posterior_state, action, observation.to(device=self.model.device), explore=True)

                    next_obs, reward, done = env.step(action[0].cpu())
                    dataset.append(observation, action.cpu(), reward, done)
                    total_reward += reward
                    observation = next_obs

                logger.log({'train_reward': total_reward}, step=i)


    def train_only_pine(self, init_agents, new_envs):
        pass

    def test_pine(self, init_agents, new_envs):
        results = {}
        for agent in init_agents:
            this_result = {}
            for env_name, task_desc in new_envs:
                e = get_env(self.config, env_name=env_name)
                obs = e.reset().to(self.device)
                visual, text = self.model.raw_to_embeds(obs, task_desc)
                visual_sim = F.cosine_similarity(visual, agent.obs_embed, dim=-1)
                text_sim = F.cosine_similarity(text[:,0,:], agent.desc_embed, dim=-1)
                this_result[env_name] = {'visual similarity' : visual_sim, 'text similarity' : text_sim}


            all_textsims = torch.stack([this_result[k]['text similarity'] for k in this_result])
            text_norm = all_textsims - all_textsims.min()
            text_norm /= text_norm.max()
            for i, k in enumerate(this_result):
                this_result[k]['text similarity normalized'] = text_norm[i]

            results[agent.task_name] = this_result if agent.task_name not in results else results[
                                                                                              agent.task_name] + this_result

        return results



    def train(self, env_list, n_iters):



        for env, task_descriptor in tqdm(env_list, desc='Training on envs'):

            logger = wandb.init(
                project='Pine',
                name=f'Pine_run_{env}',
                reinit=True
            )


            env = get_env(self.config, env_name=env)
            self.buffer = ExperienceReplay(self.config['experience size'], False, env.observation_size, env.action_size,
                                       self.config['bit depth'], self.config['device'])

            init_databuffer(self.buffer, env, self.config['seed episodes'])


            solved, confidence = self.model.new_task(env, task_descriptor)
            logger.log({'solved': solved, 'confidence': confidence})
            if solved:
               reward = self.model.test_run(5) / 5
               wandb.log({'initial_test_reward' : reward})


            else:
                reward = self.model.test_run(5) / 5
                logger.log({'pre_train_test_reward' : reward})
                self.train_new_agent(env, self.buffer, n_iters, logger)
                reward = self.model.test_run(5) / 5
                logger.log({'after_train_test_reward' : reward})
                self.model.update_agent()

            logger.finish()
            torch.save(self.model.model_save(), f'checkpoints/pine_{len(self.model.agents)}_agents.pth')






