
import torch
from torch import jit, nn
from torch.nn import functional as F


class Planner(nn.Module):

    def __init__(self, config, transition_model, reward_model):
        super(Planner, self).__init__()

        self.transition_model   = transition_model
        self.reward_model       = reward_model

        self.candidates         = config['candidates']
        self.optimize_iters     = config['optimisation iterations']
        self.planning_horizon   = config['planning horizon']
        self.top_candidates     = config['top candidates']
        self.action_size        = config['action space']
        self.max_action         = config['max action']
        self.min_action         = config['min action']



    def forward(self, belief, state):
        B, H, Z = belief.size(0), belief.size(1), state.size(1)
        belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(
            dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        action_mean, action_std_dev = torch.zeros(self.planning_horizon, B, 1, self.action_size,
                                                  device=belief.device), torch.ones(self.planning_horizon, B, 1,
                                                                                    self.action_size,
                                                                                    device=belief.device)
        for _ in range(self.optimize_iters):

            actions = (action_mean + action_std_dev * torch.randn(self.planning_horizon, B, self.candidates,
                                                                  self.action_size, device=action_mean.device)).view(
                self.planning_horizon, B * self.candidates, self.action_size)
            actions.clamp_(min=self.min_action, max=self.max_action)  # Clip action range

            beliefs, states, _, _ = self.transition_model(state, actions, belief)

            returns = self.reward_model(beliefs.view(-1, H), states.view(-1, Z)).view(self.planning_horizon, -1).sum(
                dim=0)

            _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
            topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(
                dim=1)
            best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates,
                                                             self.action_size)

            action_mean, action_std_dev = best_actions.mean(dim=2, keepdim=True), best_actions.std(dim=2,
                                                                                                   unbiased=False,
                                                                                                   keepdim=True)

        return action_mean[0].squeeze(dim=1)


class TransitionModel(nn.Module):

    def __init__(self, config):
        super(TransitionModel, self).__init__()
        self._parse_config(config)
        self._build_networks()

    def _parse_config(self, cfg):
        self.act_func       = cfg['activation function']
        self.m_std_dev      = cfg['minimum standard devation']
        self.belief_size    = cfg['belief size']
        self.state_size     = cfg['state size']
        self.embedding_size     = cfg['embedding size']
        self.action_size    = cfg['action space']
        self.hidden_size    = cfg['hidden size']


    def _build_networks(self):
        # Layers
        self.embed_state_act    = nn.Linear(self.state_size + self.action_size, self.belief_size)
        self.rnn                = nn.GRUCell(self.belief_size, self.belief_size)

        self.embed_belief_pre   = nn.Linear(self.belief_size, self.hidden_size)
        self.embed_belief_post  = nn.Linear(self.belief_size + self.embedding_size, self.hidden_size)
        self.state_pre          = nn.Linear(self.hidden_size, 2 * self.state_size)
        self.state_post          = nn.Linear(self.hidden_size, 2 * self.state_size)
        self.act_func = F.relu if self.act_func == 'relu' else None


    def forward(self, prev_state, actions, prev_belief, observations=None, nontermals=None):
        n_actions = actions.shape[0] + 1

        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = [[torch.empty(0)]*n_actions for i in range(7)]

        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state

        for t in range(n_actions -1):
            state = prior_states[t] if observations is None else posterior_states[t]
            state = state if nontermals is None else state * nontermals[t]


            p_hidden = self.embed_state_act(torch.cat([state, actions[t]], dim=1))
            hidden = self.act_func(p_hidden)
            beliefs[t+1] = self.rnn(hidden, beliefs[t])

            hidden = self.act_func(self.embed_belief_pre(beliefs[t+1]))

            prior_means[t+1], prior_std_dev = torch.chunk(self.state_pre(hidden), 2, dim=1)
            prior_std_devs[t+1] = F.softplus(prior_std_dev) + self.m_std_dev
            prior_states[t+1] = prior_means[t+1] + prior_std_devs[t+1]*torch.randn_like(prior_means[t+1])

            #Unsure if i want to keep this, i will inverstigate further once the code is complete.
            if observations is not None:

                prev_t = t-1
                hidden = self.act_func(self.embed_belief_post(torch.cat([beliefs[t+1], observations[t]], dim=1)))
                posterior_means[t+1], posterior_std_dev = torch.chunk(self.state_post(hidden), 2, dim=1)
                posterior_std_devs[t+1] = F.softplus(posterior_std_dev) + self.m_std_dev
                posterior_states[t+1]  = posterior_means[t+1] + posterior_std_devs[t+1]*torch.randn_like(posterior_means[t+1])

        hidden = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
        if observations is not None:
            hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0),
                       torch.stack(posterior_std_devs[1:], dim=0)]

        return hidden



class ObservationModel(nn.Module):

    def __init__(self, config):
        super(ObservationModel, self).__init__()
        self._parse_config(config)
        self._build_networks()

    def _parse_config(self, cfg):
        self.act_func       = cfg['activation function']
        self.belief_size    = cfg['belief size']
        self.state_size     = cfg['state size']
        self.embed_size     = cfg['embedding size']



    def _build_networks(self):
        # Layers
        self.fc1 = nn.Linear(self.belief_size + self.state_size, self.embed_size)
        self.conv1 = nn.ConvTranspose2d(self.embed_size, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

        if self.act_func != 'relu':
            raise NotImplementedError("Not implemented other than Relu as of now")

        self.act_func = F.relu


    def forward(self, belief, state):
        hidden = self.fc1(torch.cat([belief, state], dim=1))
        hidden = hidden.view(-1, self.embed_size, 1, 1)
        hidden = self.act_func(self.conv1(hidden))
        hidden = self.act_func(self.conv2(hidden))
        hidden = self.act_func(self.conv3(hidden))
        observation = self.conv4(hidden)

        return observation




class RewardModel(nn.Module):
    def __init__(self, config):
        super(RewardModel, self).__init__()
        self._parse_config(config)
        self._build_networks()

    def _parse_config(self, cfg):
        self.act_func = cfg['activation function']
        self.belief_size = cfg['belief size']
        self.state_size = cfg['state size']
        self.embed_size = cfg['embedding size']
        self.hidden_size = cfg['hidden size']

    def _build_networks(self):
        # Layers
        self.fc1 = nn.Linear(self.belief_size + self.state_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, 1)



        if self.act_func != 'relu':
            raise NotImplementedError("Not implemented other than Relu as of now")

        self.act_func = F.relu

    def forward(self, belief, state):
        hidden = self.act_func(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_func(self.fc2(hidden))
        reward = self.fc3(hidden).squeeze(dim=1)
        return reward

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self._parse_config(config)
        self._build_networks()

    def _parse_config(self, cfg):
        self.act_func = cfg['activation function']
        self.embed_size = cfg['embedding size']

    def _build_networks(self):
        # Layers
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc = nn.Identity() if self.embed_size == 1024 else nn.Linear(1024, self.embed_size)

        if self.act_func != 'relu':
            raise NotImplementedError("Not implemented other than Relu as of now")

        self.act_fn = F.relu


    def forward(self, observation):

        hdn1 = self.conv1(observation)
        hidden = self.act_fn(hdn1)
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
        return hidden


