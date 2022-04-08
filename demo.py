"""
This demo.py script allows loading a trained agent for demonstration.
The location to restore the agent can either be a local folder (--demo-path) or
a remotely accessible WANDB trial in of a project (--wandb-url).
"""
import argparse
import torch
import os
from rl_modules.rl_lang_agent import LangRLAgent
import torch
from rl_modules.rl_agent import RLAgent
import gym
import lanro
import numpy as np
from rollout import RolloutWorker
import random
from utils import get_env_params
from gym_wrapper import FrameStack, GrayScaleObservation
from omegaconf import OmegaConf
import wandb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo-length', type=int, default=10, help='the demo length')
    parser.add_argument('--demo-path', type=str, default="", help='Load experiment from local path')
    parser.add_argument(
        '--wandb-url',
        type=str,
        default='',
        help=
        "Download and run the trained agent locally, using the path WANDB schema <entity>/<project name>/<run url id>")

    demo_args = parser.parse_args()
    MODEL_LOAD_PATH = 'models/model_latest.pt'
    CONFIG_LOAD_PATH = 'omega_config.yaml'
    if demo_args.wandb_url:
        TMP_DEMO_DIR = "/tmp/hipss_demo"
        wandb_url = demo_args.wandb_url.replace("/runs", "")
        # create a temporary folder called demo to download the experiment files
        if os.path.exists(TMP_DEMO_DIR):
            os.system(f'rm -rf {TMP_DEMO_DIR}')
        os.makedirs(TMP_DEMO_DIR)
        wandb.restore(MODEL_LOAD_PATH, run_path=wandb_url, root=TMP_DEMO_DIR)
        wandb.restore(CONFIG_LOAD_PATH, run_path=wandb_url, root=TMP_DEMO_DIR)
        path = TMP_DEMO_DIR
        model_path = os.path.join(path, MODEL_LOAD_PATH)
        print(f"Using temporary folder {TMP_DEMO_DIR} to download and store experiment files.")
    else:
        path = demo_args.demo_path
        model_path = os.path.join(path, MODEL_LOAD_PATH)

    with open(os.path.join(path, CONFIG_LOAD_PATH), 'r') as f:
        cfg = OmegaConf.load(f.name)

    if 'Panda' in cfg.env_name:
        env = gym.make(cfg.env_name, render=True)
    else:
        env = gym.make(cfg.env_name)

    seed = np.random.randint(int(1e6))
    env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cfg.cuda:
        torch.cuda.manual_seed(cfg.seed)
        torch.backends.cudnn.benchmark = True

    cfg.cuda = torch.cuda.is_available()
    cfg.num_rollouts_per_mpi = 1

    env_params = get_env_params(env)

    if env_params['image_observation']:
        if cfg.gray_scale:
            env = GrayScaleObservation(env, keep_dim=True)
        if cfg.framestack > 0:
            env = FrameStack(env, cfg.framestack)

    if cfg.agent == "SAC":
        policy = RLAgent(cfg, env_params, env.compute_reward, None)
        language_conditioned = False
    elif cfg.agent == "LCSAC":
        policy = LangRLAgent(cfg, env_params, env.compute_reward, None, None)
        language_conditioned = True
    else:
        raise NotImplementedError

    policy.load(model_path)
    rollout_worker = RolloutWorker(env, policy, cfg, env_params, language_conditioned)
    successes = []
    for i in range(demo_args.demo_length):
        episodes = rollout_worker.generate_rollout(train_mode=False, animated=True)
        ep_success = np.mean([e['success'][-1] for e in episodes])
        successes.append(ep_success)

    print('Av Success Rate: {}'.format(np.mean(successes)))
