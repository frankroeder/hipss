import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.encoders import FeatureExtractor, LanguageEncoder
from networks.utils import weights_init_


class LanguageCritic(nn.Module):

    def __init__(self, cfg, env_params):
        super(LanguageCritic, self).__init__()
        input_size = env_params['obs']
        self.q_net = CriticEnsemble(cfg, env_params, input_size, observation_input=True)
        self.apply(weights_init_)

    def forward(self, state, action, instruction):
        return self.q_net(state, action, instruction)


class CriticEnsemble(nn.Module):

    def __init__(self, cfg, env_params, input_size=None, observation_input=False):
        super(CriticEnsemble, self).__init__()
        self._models = nn.ModuleList(
            [Critic(cfg, env_params, input_size, observation_input) for _ in range(cfg.n_critics)])

    def forward(self, state, action, instruction=None):
        return torch.stack([_model(state, action, instruction) for _model in self._models])


class Critic(nn.Module):

    def __init__(self, cfg, env_params, input_size, observation_input):
        super(Critic, self).__init__()
        if input_size is None:
            input_size = env_params['obs']
            input_size += 2 * env_params['goal']
        self.obs_encoder = FeatureExtractor(cfg,
                                            env_params,
                                            input_size=input_size,
                                            feature_size=cfg.feature_embedding_size,
                                            observation_input=observation_input)
        self.action_encoder = FeatureExtractor(cfg,
                                               env_params,
                                               input_size=env_params['action'],
                                               feature_size=cfg.feature_embedding_size,
                                               observation_input=False)
        embedding_size = cfg.feature_embedding_size * 2
        if 'vocab' in env_params.keys():
            self.lang_encoder = LanguageEncoder(cfg, env_params)
            embedding_size += self.lang_encoder.output_size
        self.fc1 = nn.Linear(embedding_size, cfg.hidden_size)
        self.fc2 = nn.Linear(cfg.hidden_size, 1)
        self.apply(weights_init_)

    def forward(self, obs, action, instruction=None):
        obs_embedding = self.obs_encoder(obs)
        action_embedding = self.action_encoder(action)
        embeddings = [obs_embedding, action_embedding]
        if instruction is not None:
            instruction_embedding = self.lang_encoder(instruction)
            embeddings.append(instruction_embedding)
        x = torch.cat(embeddings, dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
