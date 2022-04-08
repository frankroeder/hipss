import os
import torch
import numpy as np
from mpi_utils.mpi_utils import sync_networks
from rl_modules.buffer import ReplayBuffer
from networks import GaussianActor, CriticEnsemble
from mpi_utils.normalizer import Normalizer
from her_modules.her import HerSampler
from updates import update_flat
from utils import hard_update, soft_update, available_device


class RLAgent:

    def __init__(self, cfg, env_params, compute_rew):

        self.cfg = cfg
        self.alpha = cfg.alpha
        self.env_params = env_params

        self.total_iter = 0

        self.freq_target_update = cfg.freq_target_update

        self.actor_network = GaussianActor(cfg, env_params)
        self.critic_network = CriticEnsemble(cfg, env_params)
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)

        self.critic_target_network = CriticEnsemble(cfg, env_params)
        hard_update(self.critic_target_network, self.critic_network)
        sync_networks(self.critic_target_network)

        self.policy_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.cfg.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.cfg.lr_critic)

        self.o_norm = Normalizer(size=self.env_params['obs'], default_clip_range=self.cfg.clip_range)
        self.g_norm = Normalizer(size=self.env_params['goal'], default_clip_range=self.cfg.clip_range)

        if self.cfg.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.critic_target_network.cuda()

        self.log_alpha = None
        self.target_entropy = None
        self.alpha_optim = None
        if self.cfg.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.env_params['action'])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr_entropy)
            if self.cfg.cuda:
                self.log_alpha = self.log_alpha.cuda()

        self.her_module = HerSampler(self.cfg, compute_rew)

        self.buffer = ReplayBuffer(env_params=self.env_params,
                                   buffer_size=self.cfg.buffer_size,
                                   sample_func=self.her_module.sample_her_transitions)

    @torch.no_grad()
    def act(self, obs, ag, g, with_noise):
        input_tensor = self._preproc_inputs(obs, ag, g)
        action = self._select_actions(input_tensor, with_noise)
        return action.copy()

    def store(self, episodes):
        self.buffer.store_episode(episode_batch=episodes)

    def _preproc_inputs(self, obs, ag, g):
        obs_norm = self.o_norm.normalize(obs)
        ag_norm = self.g_norm.normalize(ag)
        g_norm = self.g_norm.normalize(g)
        inputs = np.concatenate([obs_norm, ag_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.cfg.cuda:
            inputs = inputs.cuda()
        return inputs

    def train(self):
        self.total_iter += 1
        metric_dict = self._update_network()

        if self.total_iter % self.freq_target_update == 0:
            soft_update(self.critic_target_network, self.critic_network, self.cfg.polyak)
        return metric_dict

    def _select_actions(self, state, with_noise):
        if with_noise:
            action, _, _ = self.actor_network.sample(state)
        else:
            _, _, action = self.actor_network.sample(state)
        return action.detach().cpu().numpy()[0]

    def _update_normalizer(self, episode):

        mb_obs = episode['obs']
        mb_ag = episode['ag']
        mb_g = episode['g']
        mb_actions = episode['action']
        mb_obs_next = mb_obs[1:, :]
        mb_ag_next = mb_ag[1:, :]
        num_transitions = mb_actions.shape[0]
        buffer_temp = {
            'obs': np.expand_dims(mb_obs, 0),
            'ag': np.expand_dims(mb_ag, 0),
            'g': np.expand_dims(mb_g, 0),
            'action': np.expand_dims(mb_actions, 0),
            'obs_next': np.expand_dims(mb_obs_next, 0),
            'ag_next': np.expand_dims(mb_ag_next, 0),
        }
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        self.o_norm.update(transitions['obs'])
        self.o_norm.recompute_stats()

        if self.cfg.normalize_goal:
            self.g_norm.update(transitions['g'])
            self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        g = np.clip(g, -self.cfg.clip_obs, self.cfg.clip_obs)
        if self.env_params['image_observation']:
            return np.asarray(o, dtype=np.float32), g
        o = np.clip(o, -self.cfg.clip_obs, self.cfg.clip_obs)
        return o, g

    def _update_network(self):
        transitions = self.buffer.sample(self.cfg.batch_size)

        o, o_next, g, ag, ag_next, actions, rewards = transitions['obs'], transitions['obs_next'], transitions['g'], transitions['ag'], \
                                                      transitions['ag_next'], transitions['action'], transitions['reward']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        _, transitions['ag'] = self._preproc_og(o, ag)
        _, transitions['ag_next'] = self._preproc_og(o, ag_next)

        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        ag_norm = self.g_norm.normalize(transitions['ag'])
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])

        metric_dict = update_flat(self.actor_network, self.critic_network, self.critic_target_network,
                                  self.policy_optim, self.critic_optim, self.alpha, self.log_alpha, self.target_entropy,
                                  self.alpha_optim, obs_norm, ag_norm, g_norm, obs_next_norm, actions, rewards,
                                  self.cfg)
        if 'reward_metrics' in transitions:
            metric_dict.update(transitions['reward_metrics'])
        return metric_dict

    def save(self, model_path, epoch='latest'):
        model_dict = {'actor': self.actor_network.state_dict(), 'critic': self.critic_network.state_dict()}
        torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, model_dict],
                   os.path.join(model_path, f'model_{epoch}.pt'))

    def load(self, model_path):
        if os.path.islink(model_path):
            model_path = os.readlink(model_path)
        o_mean, o_std, g_mean, g_std, model_dict = torch.load(model_path, map_location=available_device())
        self.actor_network.load_state_dict(model_dict['actor'])
        self.actor_network.eval()
        self.o_norm.mean = o_mean
        self.o_norm.std = o_std
        self.g_norm.mean = g_mean
        self.g_norm.std = g_std
