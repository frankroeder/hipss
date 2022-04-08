from collections import deque
import numpy as np
from gym.spaces import Box
from gym import ObservationWrapper


class FrameStack(ObservationWrapper):

    def __init__(self, env, num_frames):
        super(FrameStack, self).__init__(env)
        self._env = env
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)

        low = np.repeat(self.observation_space['observation'].low[np.newaxis, ...], num_frames, axis=0)
        high = np.repeat(self.observation_space['observation'].high[np.newaxis, ...], num_frames, axis=0)
        self.observation_space['observation'] = Box(low=low,
                                                    high=high,
                                                    dtype=self.observation_space['observation'].dtype)

    def observation(self):
        assert len(self.frames) == self.num_frames, (len(self.frames), self.num_frames)
        return np.stack(list(self.frames), axis=0)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation['observation'])
        return {'observation': self.observation(), 'instruction': observation['instruction']}, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation['observation']) for _ in range(self.num_frames)]
        return {'observation': self.observation(), 'instruction': observation['instruction']}

    def __getattr__(self, name):
        return getattr(self._env, name)


class GrayScaleObservation(ObservationWrapper):
    r"""Convert the image observation from RGB to gray scale."""

    def __init__(self, env, keep_dim=False):
        super(GrayScaleObservation, self).__init__(env)
        self._env = env
        self.keep_dim = keep_dim

        assert (len(env.observation_space['observation'].shape) == 3
                and env.observation_space['observation'].shape[-1] == 3)

        obs_shape = self.observation_space['observation'].shape[:2]
        if self.keep_dim:
            self.observation_space['observation'] = Box(low=0,
                                                        high=255,
                                                        shape=(obs_shape[0], obs_shape[1], 1),
                                                        dtype=np.uint8)
        else:
            self.observation_space['observation'] = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        import cv2
        observation['observation'] = cv2.cvtColor(observation['observation'], cv2.COLOR_RGB2GRAY)
        if self.keep_dim:
            observation['observation'] = np.expand_dims(observation['observation'], -1)
        return observation

    def __getattr__(self, name):
        return getattr(self._env, name)
