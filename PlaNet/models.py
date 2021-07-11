from typing import Optional, List
import torch
from torch import jit, nn, optim
from torch.nn import functional as F
from planner import MPCPlanner
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from transformers import BertModel, BertTokenizer
from env import EnvBatcher
import numpy as np
# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple):
  x_sizes = tuple(map(lambda x: x.size(), x_tuple))
  y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
  y_size = y.size()
  return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])



class LikenessNetworkSingle(nn.Module):

  def __init__(self):
    super(LikenessNetworkSingle, self).__init__()
    self.layer = nn.Linear(2048, 1)

  def forward(self, x):
    return self.layer(x)

class LikenessNetworkMultiple(nn.Module):

  def __init__(self):
    super(LikenessNetworkMultiple, self).__init__()
    self.layer1 = nn.Linear(2048, 4096)
    self.layer2 = nn.Linear(4096, 1024)
    self.layer3 = nn.Linear(1024, 1)

  def forward(self, x):

    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    x = self.layer3(x)

    return x




class Agent():

  def __init__(self, state_dicts, description_embed, observation_embedded, task_name='', action_size=0):

    self.state_dicts = state_dicts
    self.desc_embed = description_embed

    self.obs_embed = observation_embedded
    self.task_name = task_name

    self.action_size = action_size


  def add_embeddings(self, visual, text):

    self.desc_embed = torch.cat((self.desc_embed, text), dim=0)
    self.obs_embed  = torch.cat((self.obs_embed, visual), dim=0)






class Pine():

  def __init__(self, config, saved_dict=None):
    self.config = config
    if saved_dict is not None:
      self.planet = PlaNet(config, visual_networks=saved_dict['networks'])
      self.agents = saved_dict['agents']
    else:
      self.planet = PlaNet(config)
      self.agents = []

    self.encoder = self.planet.encoder
    self.active_agent = None
    self.device = config['device']
    self.dev_agents = {}

    self.max_episode_length = config['max episode length']
    self.tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    self.bert = BertModel.from_pretrained('bert-large-cased').eval().to(self.device)
    self.env = None
    self.select_threshold = 0.85

  def forward(self, observations, actions, nonterminals):
    return self.planet.forward(observations, actions, nonterminals)

  def new_task(self, env, task_description):

    self.env = env
    self.planet.env = env

    agent, confidence, o_embed, d_embed = self.find_agent(env, task_description)

    if agent is None:
      self.planet.new_task_network(self.config, env)
      return False, confidence

    self.solved = True if confidence >= self.select_threshold else False

    self.load_agent(agent, env, (o_embed, d_embed), copy=not self.solved)


    return self.solved, confidence

  def find_agent(self, env, task_description):
    # Present a new env for the network and decide if this is a known one or not
    self.env = env

    if len(self.agents) == 0:
      return None, 1


    observation = self.env.reset().to(self.device)

    obs_embed, desc_embed = self.raw_to_embeds(observation, task_description)

    obs_sim, desc_sim = self.compare_agents(obs_embed, desc_embed[:,0,:])


    agent, confidence = self.task_confidence(obs_sim, desc_sim)

    print(f'Selected agent {agent.task_name} with similarities {obs_sim}, {desc_sim}, giving a confidence of {confidence}')

    return agent, confidence, obs_embed, desc_embed


  def raw_to_embeds(self, visual, text, text_method='CLS'):
    with torch.no_grad():
      obs_embed = self.encoder(visual)
      desc_embeed = self.tokenizer(text, padding=True, return_tensors='pt').to(self.device)
      desc_embeed = self.bert(**desc_embeed)[0]

    return obs_embed, desc_embeed


  def load_agent(self, agent, new_env, embeddings, copy=False):
    if copy:
      agent = Agent(agent.state_dicts, embeddings[0], embeddings[1], action_size=agent.action_size, task_name=f'{agent.task_name}-derived {new_env.env_name}')
      self.agents.append(agent)

    else:
      agent.add_embeddings(embeddings[0], embeddings[1])

    self._load_task_models_dict(agent.state_dicts, agent.action_size)
    self.planet.set_optim(self.config)


    if agent.action_size != new_env.action_size:
      print("Warning, misaligned action sizes, most likely a mislabeled env!")
      self.planet.transition_model.change_action_size(new_action_size=new_env.action_size, device=self.device)
      agent.action_size = new_env.action_size

    self.planet.create_planner(new_env)

    self.planet.assembled = True
    self.active_agent = agent

  def update_agent(self):
    self.active_agent.state_dicts = (self.planet.transition_model.state_dict(), self.planet.reward_model.state_dict())

  def task_confidence(self, obs_similarites, desc_similarities):
      val, idx = torch.max(torch.stack(obs_similarites)*torch.stack(desc_similarities), dim=0)

      return self.agents[idx], val


  def compare_agents(self, obs_embed, description_embed, comparison=F.cosine_similarity):


    return zip(*[(comparison(obs_embed, agent.obs_embed, dim=-1).max(), comparison(description_embed, agent.desc_embed, dim=-1).max()) for agent in self.agents])



  @torch.no_grad()
  def compare_task_description(self, description):
    tokens = self.tokenizer(description, padding=True, return_tensors='pt')
    tokens = tokens.to(self.device)

    this_embed = self.bert(**tokens)[0]
    this_embed = this_embed[:,0,:].unsqueeze(0)
    cos_sim = F.cosine_similarity(this_embed, self.agent_embeddings[0], dim=2)

    return cos_sim, this_embed



  def test_run(self, test_episodes):

    self.eval()

    batch_envs = EnvBatcher(env=(self.env, test_episodes))

    with torch.no_grad():
      observation, total_rewards = batch_envs.reset(), np.zeros((test_episodes,))
      belief, posterior_state, action = torch.zeros(test_episodes, self.planet.belief_size,
                                                    device=self.device),\
                                        torch.zeros(test_episodes, self.planet.state_size,
                                                    device=self.device), \
                                        torch.zeros(test_episodes, self.env.action_size,
                                                    device=self.device)

      for t in range(self.max_episode_length // self.env.action_repeat):
        belief, posterior_state, action = self.planet.get_action(belief, posterior_state, action,
                                                                       observation.to(device=self.device),
                                                                       action_max=self.env.action_range[1],
                                                                       action_min=self.env.action_range[0])
        next_obs, reward, done = batch_envs.step(action.cpu())

        total_rewards += reward.numpy().sum()

        observation = next_obs
        if done.sum().item() == test_episodes:
          break

    return total_rewards

  def get_action(self, belief, posterior_state, action, observation, explore=False, action_max=None, action_min=None):

    return self.planet(belief, posterior_state, action, observation, explore, action_max, action_min)

  def eval(self):
    self.planet.eval()

  def train(self):
    self.planet.train()


  def model_save(self):
    d = {
      'agents' : self.agents,
      'networks' : [self.planet.encoder.state_dict(), self.planet.observation_model.state_dict()]
    }
  def model_load(self, model_dict):
    self.agents = model_dict['agents']
    self.encoder.load_state_dict(model_dict['networks'][0])
    self.planet.observation_model.load_state_dict(model_dict['networks'][1])


  def task_models_dict(self):
    return self.planet.transition_model.state_dict(), self.planet.reward_model.state_dict()

  def _load_task_models_dict(self, dicts, action_space):
    self.planet.init_task_networks(action_space, statedicts=dicts)


  def state_dict(self):
    return self.planet.state_dict()

  def load_dict(self, planet_dict):
    self.planet.load_dict(planet_dict)

  def manual_agent_add(self, statedicts, descriptors, initial_observation, action_size=0, name=''):
    with torch.no_grad():
      tokens = self.tokenizer(descriptors, padding=True, return_tensors='pt').to(self.device)
      outputs = self.bert(**tokens)

    self.agents.append(Agent(statedicts, outputs[0][:,0,:], self.planet.encoder(initial_observation.to(self.device)), action_size=action_size, task_name=name))



class PlaNet():


  def __init__(self, config, env=None, visual_networks=None):
    super(PlaNet, self).__init__()
    self.assembled = False
    self.config = config
    self.device = config['device']
    self.belief_size = config['belief size']
    self.state_size = config['state size']
    self.planning_horizon = config['planning horizon']
    self.optimisation_iters = config['optimisation iters']
    self.candidates = config['candidates']
    self.top_candidates = config['top candidates']
    self.hidden_size = config['hidden size']
    self.adam_epsilon = config['adam epsilon']
    self.embedding_size = config['embedding size']
    self.action_noise = config['action noise']
    self.env = env
    self.batch_size = config['batch size']

    self.free_nats = torch.full((1, ), config['free nats'], dtype=torch.float32, device=self.device)


    if env is None and not config['symbolic env']:
      self.init_visual_networks( (3, 64, 64))
      if visual_networks is not None:
        self.encoder.load_state_dict(visual_networks[0])
        self.observation_model.load_state_dict(visual_networks[1])

    else:
      self.init_visual_networks(env.observation_size)
      self.init_task_networks(config, env)
      self.set_optim(config)
      self.assembled = True


  def init_visual_networks(self, observation_size):
    self.encoder = Encoder(False, observation_size, self.embedding_size).to(
      device=self.device)

    self.observation_model = ObservationModel(False, observation_size, self.belief_size, self.state_size,
                                              self.embedding_size).to(device=self.device)


  def init_task_networks(self, action_size, statedicts=None):


    self.transition_model = TransitionModel(self.belief_size, self.state_size, action_size, self.hidden_size,
                                       self.embedding_size, ).to(device=self.device)

    self.reward_model = RewardModel(self.belief_size, self.state_size, self.hidden_size).to(
      device=self.device)

    if statedicts is not None:
      self.transition_model.load_state_dict(statedicts[0])
      self.reward_model.load_state_dict(statedicts[1])



  def create_planner(self, env):
    self.planner = MPCPlanner(env.action_size, self.planning_horizon, self.optimisation_iters, self.candidates,
                         self.top_candidates, self.transition_model, self.reward_model, env.action_range[0], env.action_range[1])



  def set_optim(self, config):

    self.param_list = list(self.transition_model.parameters()) + list(self.observation_model.parameters()) + list(
      self.reward_model.parameters()) + list(self.encoder.parameters())

    self.optimiser = optim.Adam(self.param_list, lr=config['learning rate'],
                                eps=self.adam_epsilon)


  def new_task_network(self, config, env):
    self.init_task_networks(config, env)
    self.create_planner(env)
    self.set_optim(config)
    self.assembled = True

  def set_task_network(self, config, env, transition_model_state_dict, reward_model_state_dict):

    self.init_task_networks(config, env, [transition_model_state_dict, reward_model_state_dict])
    self.set_optim(config)
    self.assembled = True



  def forward(self, observations, actions, nonterminals):
    if not self.assembled: raise RuntimeError('Task networks need to be initialized before doing forward pass')

    init_belief, init_state = torch.zeros(self.batch_size, self.belief_size, device=self.device),\
                              torch.zeros(self.batch_size, self.state_size, device=self.device)

    stacked_observation = bottle(self.encoder, (observations[1:],))
    x = self.transition_model(init_state, actions[:-1], init_belief, stacked_observation, nonterminals[:-1])
    beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = x
    observation = bottle(self.observation_model, (beliefs, posterior_states))
    reward = bottle(self.reward_model, (beliefs, posterior_states))
    kl_loss = torch.max(
      kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2),
      self.free_nats).mean(dim=(0, 1))

    return observation, reward, kl_loss

  def get_action(self, belief, posterior_state, action, observation, explore=False, action_max=None, action_min=None):
    belief, _, _, _, posterior_state, _, _ = self.transition_model(posterior_state, action.unsqueeze(dim=0), belief,
                                                              self.encoder(observation).unsqueeze(
                                                                dim=0))  # Action and observation need extra time dimension
    belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(
      dim=0)  # Remove time dimension from belief/state
    action = self.planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
    if explore:
      action = action + self.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
    if action_max is not None and action_min is not None: action.clamp_(min=action_min, max=action_max) # Clip action range
    else: action.clamp_(min=self.env.action_range[0], max=self.env.action_range[1])

    return belief, posterior_state, action

  def eval(self):
    self.reward_model.eval()
    self.transition_model.eval()
    self.observation_model.eval()
    self.encoder.eval()

  def train(self):
    self.reward_model.train()
    self.encoder.train()
    self.observation_model.train()
    self.transition_model.train()

  def state_dict(self):
    d = [
      self.reward_model.state_dict(),
      self.transition_model.state_dict(),
      self.observation_model.state_dict(),
      self.encoder.state_dict()
    ]

    return d

  def load_dict(self, planet_dict):
    d = planet_dict['planet_model']

    self.reward_model.load_state_dict(d[0])
    self.transition_model.load_state_dict(d[1])
    self.observation_model.load_state_dict(d[2])
    self.encoder.load_state_dict(d[3])

  def save_task_networks(self):
    return self.transition_model.state_dict(), self.reward_model.state_dict()

class TransitionModel(nn.Module):
  __constants__ = ['min_std_dev']

  def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, activation_function='relu', min_std_dev=0.1):
    super().__init__()

    self.state_size = state_size
    self.belief_size = belief_size
    self.min_std_dev = min_std_dev
    self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
    self.rnn = nn.GRUCell(belief_size, belief_size)
    self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
    self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
    self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
    self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)

  # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
  # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
  # t :  0  1  2  3  4  5
  # o :    -X--X--X--X--X-
  # a : -X--X--X--X--X-
  # n : -X--X--X--X--X-
  # pb: -X-
  # ps: -X-
  # b : -x--X--X--X--X--X-
  # s : -x--X--X--X--X--X-

  def forward(self, prev_state:torch.Tensor, actions:torch.Tensor, prev_belief:torch.Tensor, observations:Optional[torch.Tensor]=None, nonterminals:Optional[torch.Tensor]=None) -> List[torch.Tensor]:
    # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
    T = actions.size(0) + 1
    beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
    beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state
    # Loop over time sequence
    for t in range(T - 1):
      _state = prior_states[t] if observations is None else posterior_states[t]  # Select appropriate previous state
      _state = _state if nonterminals is None else _state * nonterminals[t]  # Mask if previous transition was terminal
      # Compute belief (deterministic hidden state)
      hidden = F.relu(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
      beliefs[t + 1] = self.rnn(hidden, beliefs[t])
      # Compute state prior by applying transition dynamics
      hidden = F.relu(self.fc_embed_belief_prior(beliefs[t + 1]))
      prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
      prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
      prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])     
      if observations is not None:
        # Compute state posterior by applying transition dynamics and using current observation
        t_ = t - 1  # Use t_ to deal with different time indexing for observations
        hidden = F.relu(self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)))
        posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
        posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
        posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
    # Return new hidden states
    hidden = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
    if observations is not None:
      hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0), torch.stack(posterior_std_devs[1:], dim=0)]
    return hidden

  def change_action_size(self, new_action_size, device):
    self.fc_embed_state_action = nn.Linear(self.state_size + new_action_size, self.belief_size).to(device)


class SymbolicObservationModel(jit.ScriptModule):
  def __init__(self, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
    super().__init__()

    self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, observation_size)


  def forward(self, belief, state):
    hidden = F.relu(self.fc1(torch.cat([belief, state], dim=1)))
    hidden = F.relu(self.fc2(hidden))
    observation = self.fc3(hidden)
    return observation


class VisualObservationModel(nn.Module):
  __constants__ = ['embedding_size']
  
  def __init__(self, belief_size, state_size, embedding_size, activation_function='relu'):
    super().__init__()

    self.embedding_size = embedding_size
    self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
    self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
    self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
    self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
    self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)


  def forward(self, belief, state):
    hidden = self.fc1(torch.cat([belief, state], dim=1))  # No nonlinearity here
    hidden = hidden.view(-1, self.embedding_size, 1, 1)
    hidden = F.relu(self.conv1(hidden))
    hidden = F.relu(self.conv2(hidden))
    hidden = F.relu(self.conv3(hidden))
    observation = self.conv4(hidden)
    return observation


def ObservationModel(symbolic, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
  if symbolic:
    return SymbolicObservationModel(observation_size, belief_size, state_size, embedding_size, activation_function)
  else:
    return VisualObservationModel(belief_size, state_size, embedding_size, activation_function)


class RewardModel(nn.Module):
  def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
    super().__init__()

    self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, 1)

  def forward(self, belief, state):
    hidden = F.relu(self.fc1(torch.cat([belief, state], dim=1)))
    hidden = F.relu(self.fc2(hidden))
    reward = self.fc3(hidden).squeeze(dim=1)
    return reward


class SymbolicEncoder(nn.Module):
  def __init__(self, observation_size, embedding_size, activation_function='relu'):
    super().__init__()

    self.fc1 = nn.Linear(observation_size, embedding_size)
    self.fc2 = nn.Linear(embedding_size, embedding_size)
    self.fc3 = nn.Linear(embedding_size, embedding_size)


  def forward(self, observation):
    hidden = F.relu(self.fc1(observation))
    hidden = F.relu(self.fc2(hidden))
    hidden = self.fc3(hidden)
    return hidden


class VisualEncoder(nn.Module):
  __constants__ = ['embedding_size']
  
  def __init__(self, embedding_size, activation_function='relu'):
    super().__init__()

    self.embedding_size = embedding_size
    self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
    self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
    self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)


  def forward(self, observation):
    hidden = F.relu(self.conv1(observation))
    hidden = F.relu(self.conv2(hidden))
    hidden = F.relu(self.conv3(hidden))
    hidden = F.relu(self.conv4(hidden))
    hidden = hidden.view(-1, 1024)
    hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
    return hidden


def Encoder(symbolic, observation_size, embedding_size, activation_function='relu'):
  if symbolic:
    return SymbolicEncoder(observation_size, embedding_size, activation_function)
  else:
    return VisualEncoder(embedding_size, activation_function)
