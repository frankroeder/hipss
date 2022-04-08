import os
import json
import numpy as np
import torch
from gym import spaces
from omegaconf import OmegaConf


def available_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def format_number(num):
    return '{:,}'.format(num)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def soft_update(target, source, polyak):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - polyak) * param.data + polyak * target_param.data)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class NumpyArrayEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, object):
            # NOTE: Ignore classes for now
            return None
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def init_storage(cfg):
    logdir = os.getcwd()
    model_path = os.path.join(logdir, 'models')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    video_path = os.path.join(logdir, 'videos')
    if cfg.log_video and not os.path.exists(video_path):
        os.mkdir(video_path)
    with open(os.path.join(logdir, 'omega_config.yaml'), 'w') as file:
        OmegaConf.save(config=cfg, f=file)
    return logdir, model_path


def is_image_observation(observation_space):
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        # Check the type
        if observation_space.dtype == np.uint8:
            return True
        # Check the value range
        elif np.any(observation_space.low == 0) or np.any(observation_space.high == 255):
            return True
        else:
            return False
    return False


def get_env_params(env):
    obs = env.reset()
    params = {
        'obs': obs['observation'].shape[0],
        'image_observation': is_image_observation(env.observation_space['observation']),
        'action': env.action_space.shape[0],
        'action_max': env.action_space.high[0],
        'max_timesteps': env._max_episode_steps,
    }
    # goal-conditioned environment
    if 'desired_goal' in obs.keys():
        params['goal'] = obs['desired_goal'].shape[0]
    # pixel-based observation
    if params['image_observation']:
        params['img'] = obs['observation'].shape
        params['channels'] = obs['observation'].shape[-1]
    # language-conditioned environment
    if 'instruction' in obs.keys():
        params['vocab'] = env.get_vocab()
        params['instruction'] = env.get_max_instruction_len()
    return params
