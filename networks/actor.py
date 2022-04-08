import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from networks.encoders import FeatureExtractor, LanguageEncoder
from networks.utils import weights_init_

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class GaussianActor(nn.Module):

    def __init__(self, cfg, env_params, action_space=None):
        super(GaussianActor, self).__init__()
        input_size = env_params['obs']
        self.feature_extractor = FeatureExtractor(cfg,
                                                  env_params,
                                                  input_size=input_size + 2 * env_params['goal'],
                                                  feature_size=cfg.feature_embedding_size)
        self.linear1 = nn.Linear(cfg.feature_embedding_size, cfg.hidden_size)
        self.linear2 = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.mean_linear = nn.Linear(cfg.hidden_size, env_params['action'])
        self.log_std_linear = nn.Linear(cfg.hidden_size, env_params['action'])

        self.apply(weights_init_)

        if action_space is None:
            self.register_buffer('action_scale', torch.tensor(1.))
            self.register_buffer('action_bias', torch.tensor(0.))
        else:
            self.register_buffer('action_scale', torch.FloatTensor((action_space.high - action_space.low) / 2.))
            self.register_buffer('action_bias', torch.FloatTensor((action_space.high + action_space.low) / 2.))

    def forward(self, state):
        x = self.feature_extractor(state)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class LanguageActor(nn.Module):

    def __init__(self, cfg, env_params, action_space=None):
        super(LanguageActor, self).__init__()
        observation_input = True
        input_size = env_params['obs']
        self.feature_extractor = FeatureExtractor(cfg,
                                                  env_params,
                                                  input_size=input_size,
                                                  feature_size=cfg.feature_embedding_size,
                                                  observation_input=observation_input)
        self.lang_encoder = LanguageEncoder(cfg, env_params)
        head_size = cfg.feature_embedding_size + self.lang_encoder.output_size
        self.linear = nn.Linear(head_size, cfg.hidden_size)
        self.mean_linear = nn.Linear(cfg.hidden_size, env_params['action'])
        self.log_std_linear = nn.Linear(cfg.hidden_size, env_params['action'])

        self.apply(weights_init_)

        if action_space is None:
            self.register_buffer('action_scale', torch.tensor(1.))
            self.register_buffer('action_bias', torch.tensor(0.))
        else:
            self.register_buffer('action_scale', torch.FloatTensor((action_space.high - action_space.low) / 2.))
            self.register_buffer('action_bias', torch.FloatTensor((action_space.high + action_space.low) / 2.))

    def forward(self, state, instruction):
        obs_embedding = self.feature_extractor(state)
        lang_embedding = self.lang_encoder(instruction)
        xl = torch.cat([obs_embedding, lang_embedding], 1)
        x = F.relu(self.linear(xl))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state, instruction):
        mean, log_std = self.forward(state, instruction)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)
