from datetime import datetime
import os
import random
import time
import uuid

import gym
from mpi4py import MPI
import numpy as np
import torch
import lanro
import wandb

from gym_wrapper import FrameStack, GrayScaleObservation
from mpi_utils import logger
from mpi_utils.mpi_utils import sync_networks
from rl_modules.rl_agent import RLAgent
from rl_modules.rl_lang_agent import LangRLAgent
from rollout import RolloutWorker
from utils import get_env_params, init_storage, count_parameters, format_number
from her_modules.hipss import HIPSSModule

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from gym.wrappers import RecordVideo


def check_hydra_config(cfg):
    with open_dict(cfg):
        cfg['num_workers'] = MPI.COMM_WORLD.Get_size()
        if cfg['seed'] is None:
            cfg['seed'] = np.random.randint(int(1e6))
    assert isinstance(cfg.env_name, str)
    assert isinstance(cfg.buffer_size, int)
    assert cfg.cnn_architecture in ['dqn', 'drqv2'], f"{cfg.cnn_architecture} is not a valid cnn architecture"
    assert cfg.agent in ['SAC', 'LCSAC'], f"{cfg.agent} is not a valid agent"


def init_hipss_module(cfg, env, env_params):
    if cfg.hindsight.name == 'hipss':
        hipss_module = HIPSSModule(cfg, env_params, env)
        sync_networks(hipss_module.model)
        if cfg.cuda:
            hipss_module.model.cuda()
        return hipss_module
    return None


def launch(cfg: DictConfig):
    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.info(OmegaConf.to_yaml(cfg))
    t_total_init = time.time()
    env = gym.make(cfg.env_name)
    env_params = get_env_params(env)

    if env_params['image_observation']:
        if cfg.gray_scale:
            env = GrayScaleObservation(env, keep_dim=True)
            env_params['img'] = env.observation_space['observation'].shape
            env_params['channels'] = env.observation_space['observation'].shape[-1]
        if cfg.framestack > 0:
            env = FrameStack(env, cfg.framestack)
        if hasattr(env, 'frames'):
            env_params['framestack'] = env.frames.maxlen
        else:
            env_params['framestack'] = 1

    # set random seeds for reproducibility
    rank_seed = cfg.seed + rank
    os.environ['PYTHONHASHSEED'] = str(rank_seed)
    env.seed(rank_seed)
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if cfg.cuda:
        torch.cuda.manual_seed(rank_seed)
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

    hipss_module = init_hipss_module(cfg, env, env_params)

    if cfg.agent == "SAC":
        assert 'NL' not in cfg.env_name, "Use agent=LCSAC for language-conditioned tasks"
        policy = RLAgent(cfg, env_params, env.compute_reward)
        language_conditioned = False
    elif cfg.agent == "LCSAC":
        assert 'NL' in cfg.env_name, "Use agent=SAC for goal-conditioned tasks"
        policy = LangRLAgent(cfg, env_params, env.compute_reward, hipss_module)
        language_conditioned = True
    else:
        raise NotImplementedError

    # pass policy to have access to normalizer etc.
    if hipss_module is not None:
        hipss_module.set_policy(policy)

    if rank == 0:
        print("actor parameters", format_number(count_parameters(policy.actor_network)))
        print(policy.actor_network)
        print("critic parameters", format_number(count_parameters(policy.critic_network)))
        print(policy.critic_network)
        if cfg.hindsight.name == 'hipss':
            print("hipss parameters", format_number(count_parameters(hipss_module.model)))
            print(hipss_module.model)

        logdir, model_path = init_storage(cfg)
        logger.configure(dir=logdir, format_strs=cfg.logging_formats)
        start_time = time.time()
        if cfg.wandb:
            wandb_args = dict(project=cfg.project_name if cfg.project_name else "{}_{}".format(cfg.agent, cfg.env_name),
                              name=f"trial_{str(uuid.uuid4())[:5]}",
                              config=OmegaConf.to_container(cfg),
                              reinit=False)
            if 'tensorboard' in cfg.logging_formats:
                # auto-upload tensorboard metrics
                wandb_args['sync_tensorboard'] = True
                wandb_args['monitor_gym'] = True
            if 'entity' in cfg:
                wandb_args['entity'] = cfg.entity
            if 'group' in cfg:
                wandb_args['group'] = cfg.group
            if 'tags' in cfg:
                wandb_args['tags'] = cfg.tags
            run = wandb.init(**wandb_args)
            wandb.save(os.path.join(logdir, 'omega_config.yaml'))
    rollout_worker = RolloutWorker(env, policy, cfg, env_params, language_conditioned=language_conditioned)
    for epoch in range(cfg.n_epochs):
        t_init = time.time()
        time_dict = dict(rollout=0.0,
                         store=0.0,
                         norm_update=0.0,
                         policy_train=0.0,
                         lp_update=0.0,
                         eval=0.0,
                         epoch=0.0,
                         int_module=0.0,
                         hipss_module=0.0)
        train_metrics = {}

        for _ in range(cfg.n_cycles):

            # Environment interactions
            t_i = time.time()
            train_episodes = rollout_worker.generate_rollout(train_mode=True)
            time_dict['rollout'] += time.time() - t_i
            if hipss_module is not None:
                hipss_module.store_rollout(train_episodes)

            # Storing episodes
            t_i = time.time()
            policy.store(train_episodes)
            time_dict['store'] += time.time() - t_i

            # Updating observation normalization
            if not env_params['image_observation']:
                t_i = time.time()
                for e in train_episodes:
                    policy._update_normalizer(e)
                time_dict['norm_update'] += time.time() - t_i

            # Update hipss module
            if hipss_module is not None and epoch % cfg.hindsight.train_freq == 0 and epoch > 0:
                t_i = time.time()
                hipss_metrics = hipss_module.train()
                for _key, _val in hipss_metrics.items():
                    train_metrics.setdefault(_key, []).append(_val)
                time_dict['hipss_module'] += time.time() - t_i

            # Policy updates
            t_i = time.time()
            for _ in range(cfg.n_batches):
                metric_dict = policy.train()
                for _key, _val in metric_dict.items():
                    train_metrics.setdefault(_key, []).append(_val)
            time_dict['policy_train'] += time.time() - t_i

            if hasattr(env, 'get_metrics'):
                env_metrics = env.get_metrics()

        time_dict['epoch'] += time.time() - t_init
        time_dict['total'] = time.time() - t_total_init

        if hasattr(policy.her_module, 'hindsight_ctr'):
            her_ctr = policy.her_module.hindsight_ctr

        # evaluate
        t_i = time.time()
        global_train_metrics = {}

        # start video recording
        if rank == 0 and cfg.log_video:
            rollout_worker.env = RecordVideo(rollout_worker.env,
                                             video_folder=os.path.join(logdir, 'videos'),
                                             video_length=env_params['max_timesteps'],
                                             name_prefix=f'hipss_{epoch}')
        eval_success, eval_rewards = rollout_worker.generate_test_rollout()
        # close video recording
        if rank == 0 and cfg.log_video:
            rollout_worker.env.close_video_recorder()
            rollout_worker.env = rollout_worker.env.unwrapped
            # wandb should log the videos by itself when tensorboard is not enabled
            if cfg.wandb and 'tensorboard' not in cfg.logging_formats:
                wandb.log({
                    "video":
                    # only log the last test rollout of the episode
                    wandb.Video(os.path.join(logdir, 'videos', f'hipss_{epoch}-episode-{0}.mp4'), fps=4, format="gif")
                })

        time_dict['eval'] += time.time() - t_i
        timesteps = np.sum([e['timesteps'] for e in train_episodes])

        for _key, _val in train_metrics.items():
            global_train_metrics[_key] = MPI.COMM_WORLD.allreduce(np.mean(_val), op=MPI.SUM)

        global_env_metrics = {}
        if hasattr(env, 'get_metrics'):
            for _key, _val in env_metrics.items():
                global_env_metrics[_key] = MPI.COMM_WORLD.allreduce(_val, op=MPI.SUM)

        global_time_dict = {}
        for _key, _val in time_dict.items():
            global_time_dict[_key] = MPI.COMM_WORLD.allreduce(_val, op=MPI.SUM)

        global_eval_success = MPI.COMM_WORLD.allreduce(eval_success, op=MPI.SUM)
        global_rewards = MPI.COMM_WORLD.allreduce(eval_rewards, op=MPI.SUM)
        global_timesteps = MPI.COMM_WORLD.allreduce(timesteps, op=MPI.SUM)
        global_her_ctr = MPI.COMM_WORLD.allreduce(her_ctr, op=MPI.SUM)

        if rank == 0:
            for _key, _val in global_train_metrics.items():
                global_train_metrics[_key] /= MPI.COMM_WORLD.Get_size()
            for _key, _val in global_env_metrics.items():
                global_env_metrics[_key] /= MPI.COMM_WORLD.Get_size()
            for _key, _val in global_time_dict.items():
                global_time_dict[_key] /= MPI.COMM_WORLD.Get_size()
            time_elapsed = time.time() - start_time
            current_fps = int(global_timesteps / (time_elapsed + 1e-8))
            log_data = {
                'epoch': epoch,
                'success_rate': round(global_eval_success / MPI.COMM_WORLD.Get_size(), 3),
                'reward': round(global_rewards / MPI.COMM_WORLD.Get_size(), 3),
                'timesteps': global_timesteps,
                'her_ctr': global_her_ctr // MPI.COMM_WORLD.Get_size(),
                'fps': current_fps,
                **{'time/' + key: val
                   for key, val in global_time_dict.items()},
                **{'train/' + key: val
                   for key, val in global_train_metrics.items()},
            }
            if hasattr(env, 'get_metrics'):
                log_data = {**log_data, **{'env/' + key: val for key, val in global_env_metrics.items()}}
            if cfg.wandb:
                wandb.log(log_data)
            {logger.logkv(_k, _v) for _k, _v in log_data.items()}
            logger.dumpkvs()
            data_str = ' '.join([
                f'{key}: {val}' for key, val in log_data.items()
                if key in ['epoch', 'success_rate', 'reward', 'timesteps', 'fps']
            ])
            logger.info(f'[{datetime.now()}] ' + data_str)
            # Saving policy models
            if epoch % cfg.save_freq == 0:
                policy.save(model_path, epoch)
                if cfg.wandb:
                    wandb.save(os.path.join(model_path, f'model_{epoch}.pt'), base_path=os.path.split(model_path)[0])

    if rank == 0:
        policy.save(model_path)
        if cfg.wandb:
            wandb.save(os.path.join(model_path, 'model_latest.pt'), base_path=os.path.split(model_path)[0])
            wandb.save(os.path.join(logdir, 'progress.csv'))
            run.finish()

    return round(global_eval_success / MPI.COMM_WORLD.Get_size(), 3)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    check_hydra_config(cfg)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    launch(cfg)


if __name__ == '__main__':
    main()
