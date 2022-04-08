import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from mpi_utils.mpi_utils import sync_grads
from networks import FeatureExtractor
from torch.nn.utils.clip_grad import clip_grad_norm_
from lanro.utils import SHAPES


class HIPSSEncoder(nn.Module):

    def __init__(self, cfg, env_params):
        super(HIPSSEncoder, self).__init__()
        input_size = env_params['obs']
        hidden_size = cfg.hindsight.hidden_size
        num_layers = cfg.hindsight.n_layers
        dropout = cfg.hindsight.dropout
        self.obs_encoder = FeatureExtractor(cfg, env_params, input_size, feature_size=hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, x, lens=None):
        x = self.obs_encoder(x)
        if lens is not None:
            x = pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(x)
        return hidden


class HIPSSDecoder(nn.Module):

    def __init__(self, cfg, env_params):
        super(HIPSSDecoder, self).__init__()
        hidden_size = cfg.hindsight.hidden_size
        num_layers = cfg.hindsight.n_layers
        vocab_size = len(env_params['vocab'])
        dropout = cfg.hindsight.dropout
        self.gru = nn.GRU(vocab_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x, hidden = self.gru(x, hidden)
        return self.fc(x), hidden


class HIPSS(nn.Module):

    def __init__(self, cfg, env_params):
        super(HIPSS, self).__init__()
        self.vocab = env_params['vocab']
        self.vocab_size = len(env_params['vocab'])
        self.encoder = HIPSSEncoder(cfg, env_params)
        self.decoder = HIPSSDecoder(cfg, env_params)

    def encode(self, x, lens=None):
        return self.encoder(x.float(), lens=lens)

    def decode(self, x, hidden=None):
        return self.decoder(x.float(), hidden=hidden)

    def forward(self, input, target, lens=None, teacher_force_ratio=0.5):
        batch_size = input.size(0)
        target_len = target.size(1)
        outputs = torch.zeros(batch_size, target_len, self.vocab_size).to(input.device)
        # encoder hidden state as initial state for the decoder
        hidden = self.encode(input, lens=lens)
        x = target[:, 0].unsqueeze(1)
        for t in range(0, target_len):
            logits, hidden = self.decode(x, hidden=hidden)
            outputs[:, t:t + 1] = logits
            best_guess_onehot = F.one_hot(logits.argmax(-1), self.vocab_size)
            # apply teacher forcing by using the next input as target
            x = target[:, t].unsqueeze(1) if np.random.rand() < teacher_force_ratio else best_guess_onehot
        return outputs


class HIPSSModule:
    good_enough = False

    def __init__(self, cfg, env_params, env):
        self.cfg = cfg
        self.env = env
        self.batch_size = cfg.hindsight.batch_size
        self.buffer = []
        self.val_buffer = []
        self.max_intruction_len = env_params['instruction']
        self.vocab = env_params['vocab']
        self.vocab_size = len(env_params['vocab'])
        self.model = HIPSS(cfg, env_params)
        self.hipss_optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.hindsight.lr)
        pad_idx = env_params['vocab']('<pad>')
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def set_policy(self, policy):
        self.policy = policy

    def store_rollout(self, episode_data):
        for mpi_rollout in episode_data:
            if np.sum(mpi_rollout['success']):
                success_indices = np.where(mpi_rollout['success'] == True)[0]
                # take the last transition with a sparse reward as end of sequence
                _max_idx = success_indices[-1]
                _obs = mpi_rollout['obs'][0:_max_idx].copy()
                # pad sequence with zeros
                _diff = len(mpi_rollout['success']) - len(_obs)
                _obs = np.vstack((_obs, [np.zeros_like(_obs[0]) for _ in range(_diff)]))
                data_sample = [_obs, mpi_rollout['instruction'][0].copy(), _obs.shape[0]]
                if np.random.rand() < self.cfg.hindsight.val_ratio:
                    self.val_buffer.append(data_sample)
                else:
                    self.buffer.append(data_sample)

    @torch.no_grad()
    def get_instruction(self, obs):
        self.model.eval()
        obs = self.policy.o_norm.normalize(obs)
        traj = torch.from_numpy(obs).float()
        if self.cfg.cuda:
            traj = traj.cuda()
        y_hat = self.model(traj,
                           torch.zeros(traj.size(0), self.max_intruction_len, self.vocab_size).to(traj.device),
                           teacher_force_ratio=0.0)
        word_indices = F.softmax(y_hat, dim=-1).argmax(-1)
        intr_indices = []
        for w_idx_lst in word_indices:
            intr_indices.append([w_idx_lst.tolist()] * (obs.shape[1]))
        return np.array(intr_indices)[:, -1]

    def get_batch(self, buffer):
        batch_idx = np.random.choice(len(buffer), min(self.batch_size, len(buffer)))
        batch_x = torch.from_numpy(np.array([buffer[i][0] for i in batch_idx]))
        labels = torch.from_numpy(np.array([buffer[i][1] for i in batch_idx]))
        batch_y = F.one_hot(labels, self.vocab_size).float()
        seq_len = torch.from_numpy(np.array([buffer[i][2] for i in batch_idx]))
        batch_x = self.policy.o_norm.normalize(batch_x)
        if self.cfg.cuda:
            return batch_x.cuda(), batch_y.cuda(), seq_len
        else:
            return batch_x, batch_y, seq_len

    def calculate_loss(self, preds, targets):
        target_indices = targets.argmax(-1)
        flat_targets = target_indices.view(-1)
        flat_preds = preds.view(-1, self.vocab_size)
        for idx in range(len(flat_targets)):
            target_word = self.vocab.idx_to_word(flat_targets[idx].item())
            pred_word = self.vocab.idx_to_word(flat_preds[idx].argmax().item())
            # NOTE: if the target is a shape specific word, any prediction from the
            # list of possible object_words is a positive prediction
            if self.env.task.mode == 'default':
                object_words = np.array(SHAPES.SQUARE.value[1])
                # is a relevant target word
                if target_word in object_words and pred_word in object_words:
                    # change the target to the predicted word
                    flat_targets[idx] = flat_preds[idx].argmax().clone()
            elif 'shape' in self.env.task.mode:
                object_words_by_shape = np.array([shape.value[1] for shape in SHAPES])
                for obj_ws in object_words_by_shape:
                    if target_word in obj_ws and pred_word in obj_ws:
                        # change the target to the predicted word of the corresponding shape
                        flat_targets[idx] = flat_preds[idx].argmax().clone()
        return self.criterion(flat_preds, flat_targets)

    def get_acc(self, preds, targets):
        pred_indices = preds.argmax(-1)
        target_indices = targets.argmax(-1)
        for b_idx in range(target_indices.shape[0]):
            for idx in range(len(target_indices[b_idx])):
                target_word = self.vocab.idx_to_word(target_indices[b_idx][idx].item())
                pred_word = self.vocab.idx_to_word(pred_indices[b_idx][idx].item())
                if self.env.task.mode == 'default':
                    object_words = np.array(SHAPES.SQUARE.value[1])
                    if target_word in object_words and pred_word in object_words:
                        target_indices[b_idx][idx] = pred_indices[b_idx][idx].clone()
                elif 'shape' in self.env.task.mode:
                    object_words_by_shape = np.array([shape.value[1] for shape in SHAPES])
                    for obj_ws in object_words_by_shape:
                        if target_word in obj_ws and pred_word in obj_ws:
                            # change the target to the predicted word of the corresponding shape
                            target_indices[b_idx][idx] = pred_indices[b_idx][idx].clone()
        return (pred_indices.flatten() == target_indices.flatten()).float().mean().item()

    def train(self):
        metrics = {
            "hipss_train_buff": len(self.buffer),
            "hipss_val_buff": len(self.val_buffer),
        }
        self.model.train()
        losses = []
        accs = []
        for _ in range(self.cfg.hindsight.train_epochs):
            batch_x, batch_y, seq_len = self.get_batch(self.buffer)
            y_hat = self.model(batch_x, batch_y, lens=seq_len)
            self.hipss_optimizer.zero_grad()
            loss = self.calculate_loss(y_hat, batch_y)
            loss.backward()
            if self.cfg.hindsight.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), self.cfg.hindsight.max_norm)
            sync_grads(self.model)
            self.hipss_optimizer.step()
            losses.append(loss.item())
            accs.append(self.get_acc(y_hat, batch_y))

        return {
            **metrics,
            'hipss_train_loss': np.mean(losses),
            "hipss_train_acc": np.mean(accs),
            **self.eval(),
        }

    @torch.no_grad()
    def eval(self):
        self.model.eval()
        val_losses = []
        accs = []
        for _ in range(5):
            batch_x, batch_y, seq_len = self.get_batch(self.val_buffer)
            y_hat = self.model(batch_x, batch_y, lens=seq_len)
            loss = self.calculate_loss(y_hat, batch_y)
            val_losses.append(loss.item())
            accs.append(self.get_acc(y_hat, batch_y))
        self.good_enough = np.mean(accs) > self.cfg.hindsight.val_acc_threshold
        #  if self.good_enough:
        #      print("Prediction:", self.env.env.decode_instruction(y_hat[0].argmax(1).detach().numpy()))
        #      print("Target:", self.env.env.decode_instruction(batch_y[0].argmax(1).numpy()))
        self.model.train()
        return {'hipss_val_loss': np.mean(val_losses), "hipss_val_acc": np.mean(accs)}
