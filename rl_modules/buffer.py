import torch
import threading
import numpy as np


class Buffer(object):

    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._size = 0
        self._cursor = 0

    def _inc_cursor(self):
        # Increase size until max size is reached
        if self._size < self.capacity:
            self._size += 1
        # When cursor reaches end, restart at beginning, overwriting oldest entries first
        self._cursor = (self._cursor + 1) % self.capacity

    def reset(self):
        self._cursor = 0
        self._size = 0

    @property
    def size(self):
        return self._size

    def __len__(self):
        return self.size


class ErrorBuffer(Buffer):

    def __init__(self, max_size):
        super(ErrorBuffer, self).__init__(max_size, 0)
        self._errors = torch.empty(max_size, dtype=torch.float32, device=self.device)

    def _add_single(self, err):
        self._errors[self._cursor] = err
        self._inc_cursor()

    def add(self, errors):
        for _err in errors:
            self._add_single(_err)

    def get_all(self):
        return self._errors[:self._size]

    def get_min(self):
        return self.get_all().min()

    def get_max(self):
        return self.get_all().max()


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
