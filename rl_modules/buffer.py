import threading
import numpy as np


class ReplayBuffer:

    def __init__(self, env_params, buffer_size, sample_func, lang_mode=False, hipss_module=None):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        self.lang_mode = lang_mode
        self.hipss_module = hipss_module

        self.sample_func = sample_func
        self.current_size = 0

        if env_params['image_observation']:
            if env_params['framestack'] > 1:
                obs_storage = np.empty([self.size, self.T + 1, env_params['framestack'], *self.env_params['img']],
                                       dtype=np.uint8)
            else:
                obs_storage = np.empty([self.size, self.T + 1, *self.env_params['img']], dtype=np.uint8)
        else:
            obs_storage = np.empty([self.size, self.T + 1, self.env_params['obs']])

        self.buffer = {'obs': obs_storage, 'action': np.empty([self.size, self.T, self.env_params['action']])}
        if lang_mode:
            self.buffer['instruction'] = np.empty([self.size, self.T, self.env_params['instruction']], dtype=np.int64)
            self.buffer['hindsight_instruction'] = np.empty([self.size, self.T, self.env_params['instruction']],
                                                            dtype=np.int64)
            self.buffer['reward'] = np.empty([self.size, self.T, 1])
        else:
            self.buffer['ag'] = np.empty([self.size, self.T + 1, self.env_params['goal']])
            self.buffer['g'] = np.empty([self.size, self.T, self.env_params['goal']])

        self.lock = threading.Lock()

    def store_episode(self, episode_batch):
        batch_size = len(episode_batch)
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)

            for i, e in enumerate(episode_batch):
                self.buffer['obs'][idxs[i]] = e['obs']
                self.buffer['action'][idxs[i]] = e['action']

                if self.lang_mode:
                    self.buffer['instruction'][idxs[i]] = e['instruction']
                    self.buffer['hindsight_instruction'][idxs[i]] = e['hindsight_instruction']
                    self.buffer['reward'][idxs[i]] = e['reward']
                else:
                    self.buffer['ag'][idxs[i]] = e['ag']
                    self.buffer['g'][idxs[i]] = e['g']

    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffer.keys():
                temp_buffers[key] = self.buffer[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        if not self.lang_mode:
            temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        if self.lang_mode:
            transitions = self.sample_func(temp_buffers, batch_size, self.hipss_module)
        else:
            transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = [idx[0]]
        return idx
