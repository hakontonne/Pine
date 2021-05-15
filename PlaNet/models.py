from typing import Optional, List
import torch
from torch import jit, nn, optim
from torch.nn import functional as F
from planner import MPCPlanner
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence


# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple):
  x_sizes = tuple(map(lambda x: x.size(), x_tuple))
  y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
  y_size = y.size()
  return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])



class Pine():

  def __init__(self, config):
    self.config = config
    self.planet = PlaNet(config)
    self.agents = []
    self.dev_agents = {}

    self.env = None


  def forward(self, observations, actions, nonterminals):
    return self.planet.forward(observations, actions, nonterminals)

  def new_agent(self, env):

    if self.planet.assembled:
      self.dev_agents[self.env.env_name] = self.planet.save_task_networks()

    self.env = env

    self.planet.init_task_networks(self.config, env, statedicts=self.dev_agents[env.env_name] if env.env_name in self.dev_agents else None)
    self.planet.set_optim(self.config)
    self.planet.assembled = True
    self.planet.env = env

  def new_env(self, env, task_description):
    # Present a new env for the network and decide if this is a known one or not
    self.env = env

    if len(self.agents) == 0:
      # we have no agents, the pine model is completly fresh
      self.planet.init_task_networks(self.config, env)
      self.planet.set_optim(self.config)

      return self.planet


    observation = self.env.reset()
    retval = self.compare_initial_observation(observation)
    retval2 = self.compare_task_description(task_description)

    return self.eval_likeness(retval, retval2)

  def eval_likeness(self, obs_compare, descript_compare):
    pass

  def compare_initial_observation(self, observation):
    pass

  def compare_task_description(self, description):
    pass



  def get_action(self, belief, posterior_state, action, observation, explore=False, action_max=None, action_min=None):

    return self.planet(belief, posterior_state, action, observation, explore, action_max, action_min)

  def eval(self):
    self.planet.eval()

  def train(self):
    self.planet.train()

  def task_models_dict(self):
    return self.planet.transition_model.state_dict(), self.planet.reward_model.state_dict()

  def load_task_models_dict(self, dicts):
    self.planet.init_task_networks(self.config, self.env, dicts)


  def state_dict(self):
    return self.planet.state_dict()

  def load_dict(self, planet_dict):
    self.planet.load_dict(planet_dict)




class PlaNet():


  def __init__(self, config, env=None):
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


  def init_task_networks(self, config, env, statedicts=None):


    self.transition_model = TransitionModel(self.belief_size, self.state_size, env.action_size, self.hidden_size,
                                       self.embedding_size, ).to(device=self.device)

    self.reward_model = RewardModel(self.belief_size, self.state_size, self.hidden_size).to(
      device=self.device)

    if statedicts is not None:
      self.transition_model.load_state_dict(statedicts[0])
      self.reward_model.load_state_dict(statedicts[1])



    self.planner = MPCPlanner(env.action_size, self.planning_horizon, self.optimisation_iters, self.candidates,
                         self.top_candidates, self.transition_model, self.reward_model, env.action_range[0], env.action_range[1])



  def set_optim(self, config):

    self.param_list = list(self.transition_model.parameters()) + list(self.observation_model.parameters()) + list(
      self.reward_model.parameters()) + list(self.encoder.parameters())

    self.optimiser = optim.Adam(self.param_list, lr=config['learning rate'],
                                eps=self.adam_epsilon)


  def new_task_network(self, config, env):
    self.init_task_networks(config, env)
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
