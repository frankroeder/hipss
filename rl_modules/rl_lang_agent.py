import os
import torch
import numpy as np
from mpi_utils.mpi_utils import sync_networks
from rl_modules.buffer import ReplayBuffer
from networks import LanguageCritic, LanguageActor
from mpi_utils.normalizer import Normalizer
from her_modules.her import HerSampler
from updates import update_language
from utils import hard_update, soft_update, available_device


class LangRLAgent:

    def __init__(self, cfg, env_params, compute_rew, hipss_module):

        self.cfg = cfg
        self.alpha = cfg.alpha
        self.env_params = env_params
        self.hipss_module = hipss_module

        self.total_iter = 0

        self.freq_target_update = cfg.freq_target_update

        self.actor_network = LanguageActor(cfg, env_params)
        self.critic_network = LanguageCritic(cfg, env_params)
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)

        self.critic_target_network = LanguageCritic(cfg, env_params)
        hard_update(self.critic_target_network, self.critic_network)
        sync_networks(self.critic_target_network)

        self.policy_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.cfg.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.cfg.lr_critic)

        self.o_norm = Normalizer(size=self.env_params['obs'], default_clip_range=self.cfg.clip_range)

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
                                   sample_func=self.her_module.sample_her_lang_transitions,
                                   hipss_module=self.hipss_module,
                                   lang_mode=True)

    @torch.no_grad()
    def act(self, obs, instruction, with_noise):
        input_tensor, instr_tensor = self._preproc_inputs(obs, instruction)
        action = self._select_actions(input_tensor, instr_tensor, with_noise)
        return action.copy()

    def store(self, episodes):
        self.buffer.store_episode(episode_batch=episodes)

    def _preproc_inputs(self, obs, instruction):
        if self.env_params['image_observation']:
            obs_norm = obs
        else:
            obs_norm = self.o_norm.normalize(obs)
        inputs = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
        instr_tensor = torch.tensor(instruction, dtype=torch.long).unsqueeze(0)
        if self.cfg.cuda:
            inputs = inputs.cuda()
            instr_tensor = instr_tensor.cuda()
        return inputs, instr_tensor

    def train(self):
        self.total_iter += 1
        metric_dict = self._update_network()

        if self.total_iter % self.freq_target_update == 0:
            soft_update(self.critic_target_network, self.critic_network, self.cfg.polyak)
        return metric_dict

    def _select_actions(self, state, instruction, with_noise):
        if with_noise:
            action, _, _ = self.actor_network.sample(state, instruction)
        else:
            _, _, action = self.actor_network.sample(state, instruction)
        return action.detach().cpu().numpy()[0]

    def _update_normalizer(self, episode):
        mb_obs = episode['obs']
        mb_actions = episode['action']
        mb_instructions = episode['instruction']
        mb_rewards = episode['reward']
        mb_hindsight_instructions = episode['hindsight_instruction']
        mb_obs_next = mb_obs[1:, :]
        num_transitions = mb_actions.shape[0]
        buffer_temp = {
            'obs': np.expand_dims(mb_obs, 0),
            'action': np.expand_dims(mb_actions, 0),
            'obs_next': np.expand_dims(mb_obs_next, 0),
            'instruction': np.expand_dims(mb_instructions, 0),
            'reward': np.expand_dims(mb_rewards, 0),
            'hindsight_instruction': np.expand_dims(mb_hindsight_instructions, 0)
        }
        transitions = self.her_module.sample_her_lang_transitions(buffer_temp, num_transitions, None)
        obs = transitions['obs']
        transitions['obs'] = self._preproc_o(obs)
        self.o_norm.update(transitions['obs'])
        self.o_norm.recompute_stats()

    def _preproc_o(self, o):
        if self.env_params['image_observation']:
            return np.asarray(o, dtype=np.float32)
        else:
            return np.clip(o, -self.cfg.clip_obs, self.cfg.clip_obs)

    def _update_network(self):
        transitions = self.buffer.sample(self.cfg.batch_size)

        o, o_next, instruction,  actions, rewards = transitions['obs'], transitions['obs_next'], transitions['instruction'], \
            transitions['action'], transitions['reward']
        transitions['obs'] = self._preproc_o(o)
        transitions['obs_next'] = self._preproc_o(o_next)

        if self.env_params['image_observation']:
            obs_norm = transitions['obs']
            obs_next_norm = transitions['obs_next']
        else:
            obs_norm = self.o_norm.normalize(transitions['obs'])
            obs_next_norm = self.o_norm.normalize(transitions['obs_next'])

        metric_dict = update_language(self.actor_network, self.critic_network, self.critic_target_network,
                                      self.policy_optim, self.critic_optim, self.alpha, self.log_alpha,
                                      self.target_entropy, self.alpha_optim, obs_norm, instruction, obs_next_norm,
                                      actions, rewards, self.cfg)
        if 'reward_metrics' in transitions:
            metric_dict.update(transitions['reward_metrics'])
        return metric_dict

    def save(self, model_path, epoch='latest'):
        model_dict = {'actor': self.actor_network.state_dict(), 'critic': self.critic_network.state_dict()}
        torch.save([self.o_norm.mean, self.o_norm.std, model_dict], os.path.join(model_path, f'model_{epoch}.pt'))

    def load(self, model_path):
        if os.path.islink(model_path):
            model_path = os.readlink(model_path)
        o_mean, o_std, model_dict = torch.load(model_path, map_location=available_device())
        self.actor_network.load_state_dict(model_dict['actor'])
        self.actor_network.eval()
        self.o_norm.mean = o_mean
        self.o_norm.std = o_std
