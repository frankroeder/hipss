import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from mpi_utils.mpi_utils import sync_grads


def update_entropy(alpha, log_alpha, target_entropy, log_pi, alpha_optim, cfg):
    if cfg.automatic_entropy_tuning:
        alpha_loss = -(log_alpha * (log_pi + target_entropy).detach()).mean()

        alpha_optim.zero_grad()
        alpha_loss.backward()
        alpha_optim.step()

        alpha = log_alpha.exp()
        alpha_tlogs = alpha.clone()
    else:
        alpha_loss = torch.tensor(0.)
        alpha_tlogs = torch.tensor(alpha)

    return alpha_loss, alpha_tlogs


def update_flat(actor_network, critic_network, critic_target_network, policy_optim, critic_optim, alpha, log_alpha,
                target_entropy, alpha_optim, obs_norm, ag_norm, g_norm, obs_next_norm, actions, rewards, cfg):
    inputs_norm = np.concatenate([obs_norm, ag_norm, g_norm], axis=1)
    inputs_next_norm = np.concatenate([obs_next_norm, ag_norm, g_norm], axis=1)

    inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
    inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.float32)
    r_tensor = torch.tensor(rewards, dtype=torch.float32).reshape(rewards.shape[0], 1)

    if cfg.cuda:
        inputs_norm_tensor = inputs_norm_tensor.cuda()
        inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
        actions_tensor = actions_tensor.cuda()
        r_tensor = r_tensor.cuda()

    with torch.no_grad():
        actions_next, log_pi_next, _ = actor_network.sample(inputs_next_norm_tensor)
        qf_next_target = critic_target_network(inputs_next_norm_tensor, actions_next)
        min_qf_next_target = torch.min(qf_next_target, dim=0).values - alpha * log_pi_next
        next_q_value = r_tensor + cfg.gamma * min_qf_next_target

    # the q loss
    qf = critic_network(inputs_norm_tensor, actions_tensor)
    qf_loss = torch.stack([F.mse_loss(_qf, next_q_value) for _qf in qf]).mean()
    # the actor loss
    pi, log_pi, _ = actor_network.sample(inputs_norm_tensor)
    qf_pi = critic_network(inputs_norm_tensor, pi)
    min_qf_pi = torch.min(qf_pi, dim=0).values
    policy_loss = ((alpha * log_pi) - min_qf_pi).mean()

    # update actor network
    policy_optim.zero_grad()
    policy_loss.backward()
    sync_grads(actor_network)
    policy_optim.step()

    # update the critic_network
    critic_optim.zero_grad()
    qf_loss.backward()
    if cfg.clip_grad_norm:
        clip_grad_norm_(critic_network.parameters(), cfg.max_norm)
    sync_grads(critic_network)
    critic_optim.step()

    alpha_loss, alpha_tlogs = update_entropy(alpha, log_alpha, target_entropy, log_pi, alpha_optim, cfg)

    train_metrics = dict(q_loss=qf_loss.item(),
                         next_q=next_q_value.mean().item(),
                         policy_loss=policy_loss.item(),
                         alpha_loss=alpha_loss.item(),
                         alpha_tlogs=alpha_tlogs.item())
    for idx, (_qf, _qtarget) in enumerate(zip(qf, qf_next_target)):
        train_metrics[f'q_{idx}'] = _qf.mean().item()
        train_metrics[f'q_target_{idx}'] = _qtarget.mean().item()
    return train_metrics


def update_language(actor_network, critic_network, critic_target_network, policy_optim, critic_optim, alpha, log_alpha,
                    target_entropy, alpha_optim, obs_norm, instruction, obs_next_norm, actions, rewards, cfg):

    inputs_norm = obs_norm
    inputs_next_norm = obs_next_norm

    inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
    inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.float32)
    r_tensor = torch.tensor(rewards, dtype=torch.float32).reshape(rewards.shape[0], 1)
    instruction_tensor = torch.tensor(instruction, dtype=torch.long)

    if cfg.cuda:
        inputs_norm_tensor = inputs_norm_tensor.cuda()
        inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
        actions_tensor = actions_tensor.cuda()
        r_tensor = r_tensor.cuda()
        instruction_tensor = instruction_tensor.cuda()

    with torch.no_grad():
        actions_next, log_pi_next, _ = actor_network.sample(inputs_next_norm_tensor, instruction_tensor)
        qf_next_target = critic_target_network(inputs_next_norm_tensor, actions_next, instruction_tensor)
        min_qf_next_target = torch.min(qf_next_target, dim=0).values - alpha * log_pi_next
        next_q_value = r_tensor + cfg.gamma * min_qf_next_target

    # the q loss
    qf = critic_network(inputs_norm_tensor, actions_tensor, instruction_tensor)
    qf_loss = torch.stack([F.mse_loss(_qf, next_q_value) for _qf in qf]).mean()

    # the actor loss
    pi, log_pi, _ = actor_network.sample(inputs_norm_tensor, instruction_tensor)
    qf_pi = critic_network(inputs_norm_tensor, pi, instruction_tensor)
    min_qf_pi = torch.min(qf_pi, dim=0).values
    policy_loss = ((alpha * log_pi) - min_qf_pi).mean()

    # update actor network
    policy_optim.zero_grad()
    policy_loss.backward()
    sync_grads(actor_network)
    policy_optim.step()

    # update the critic_network
    critic_optim.zero_grad()
    qf_loss.backward()
    if cfg.clip_grad_norm:
        clip_grad_norm_(critic_network.parameters(), cfg.max_norm)
    sync_grads(critic_network)
    critic_optim.step()

    alpha_loss, alpha_tlogs = update_entropy(alpha, log_alpha, target_entropy, log_pi, alpha_optim, cfg)

    train_metrics = dict(q_loss=qf_loss.item(),
                         next_q=next_q_value.mean().item(),
                         policy_loss=policy_loss.item(),
                         alpha_loss=alpha_loss.item(),
                         alpha_tlogs=alpha_tlogs.item())
    for idx, (_qf, _qtarget) in enumerate(zip(qf, qf_next_target)):
        train_metrics[f'q_{idx}'] = _qf.mean().item()
        train_metrics[f'q_target_{idx}'] = _qtarget.mean().item()
    return train_metrics
