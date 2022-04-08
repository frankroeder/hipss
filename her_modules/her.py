import numpy as np


class HerSampler:

    def __init__(self, cfg, reward_func=None):
        self.replay_strategy = cfg.hindsight.replay_strategy
        self.replay_k = cfg.hindsight.replay_k
        self.method_name = cfg.hindsight.name
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + self.replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

        self.hindsight_ctr = 0

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        assert self.method_name == 'her'
        T = episode_batch['action'].shape[1]
        rollout_batch_size = episode_batch['action'].shape[0]
        batch_size = batch_size_in_transitions

        # select episodes and transitions
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}

        # select her indices for the batch of transitions
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        self.hindsight_ctr += her_indexes[0].size
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]

        # replace goal with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # re-compute reward
        transitions['reward'] = np.expand_dims(
            np.array(
                [self.reward_func(ag_next, g, None) for ag_next, g in zip(transitions['ag_next'], transitions['g'])]),
            1)
        return transitions

    def sample_her_lang_transitions(self, episode_batch, batch_size_in_transitions, hipss_module):
        T = episode_batch['action'].shape[1]
        rollout_batch_size = episode_batch['action'].shape[0]
        batch_size = batch_size_in_transitions
        # select episodes and transitions
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)

        # select episodes with hindsight instruction transitions
        hi_ep_t_indices = np.sum(episode_batch["hindsight_instruction"], -1) > 0

        # check if episodes contains HI transitions
        if np.sum(hi_ep_t_indices) and (
            (self.method_name == 'hipss' and hipss_module is not None and hipss_module.good_enough) or
            (self.method_name == 'heir')):
            # get indices of the hindsight episodes and transitions
            hi_ep_indices, hi_t_indices = np.where(hi_ep_t_indices == True)
            assert np.all(hi_ep_indices < rollout_batch_size), "hindsight episode index higher than rollout batch size"

            # replace instructions where the environment returned a hindsight instruction
            # with HIPSS predictions or the expert samples
            if self.method_name == 'hipss':
                # slicing reduces the (GPU) memory usage of HIPSS
                # and we only care about input sequences with hindsight signal from environment
                hinstr = hipss_module.get_instruction(episode_batch['obs'][hi_ep_indices, :T])
            elif self.method_name == 'heir':
                hinstr = episode_batch['hindsight_instruction'][hi_ep_t_indices]

            self.hindsight_ctr += hinstr.shape[0]
            # select episode indices `replay_k` times
            hi_ep_indices_repeat = hi_ep_indices.repeat(self.replay_k).reshape(-1, self.replay_k).T
            if self.replay_strategy == 'episode':
                # randomly sample hindsight transitions encountered within the episode before the environment signal
                new_hi_t_indices = np.random.randint(
                    low=np.zeros_like(hi_t_indices),
                    high=hi_t_indices + 1, # to include the last transitions
                    size=(self.replay_k, len(hi_t_indices)))
                assert np.all(new_hi_t_indices <= hi_t_indices.max())
                hi_t_indices = new_hi_t_indices
            elif self.replay_strategy == 'future':
                # sample future transitions coming from the same episode but after the hindsight signal
                future_idx_limit = np.ones_like(hi_t_indices) * T
                new_hi_t_indices = np.random.randint(low=hi_t_indices,
                                                     high=future_idx_limit,
                                                     size=(self.replay_k, len(hi_t_indices)))
                assert np.all(new_hi_t_indices <= T)
                hi_t_indices = new_hi_t_indices
            elif self.replay_strategy == 'final':
                # use the final and the `replay_k` transitions before the hindsight signal
                _final_her_t_idxs = hi_t_indices.copy()
                for i in range(1, self.replay_k):
                    _final_her_t_idxs = np.concatenate((_final_her_t_idxs, np.clip(hi_t_indices - i, 0, T)))
                new_hi_t_indices = _final_her_t_idxs.reshape(self.replay_k, len(hi_t_indices))

            assert hi_ep_indices_repeat.shape[0] == new_hi_t_indices.shape[0]
            for ep_idx in hi_ep_indices_repeat:
                for idx, t_idx in enumerate(new_hi_t_indices):
                    # replace instructions for hindsight transitions
                    episode_batch['instruction'][ep_idx, t_idx] = hinstr
                    # sparse reward for final transitions
                    if idx == 0 and self.replay_strategy == 'final':
                        # replace penalty with sparse reward
                        episode_batch['reward'][ep_idx, t_idx] = [0.0]
                    # otherwise reduced penalty, as those are states nearby the
                    # goal or are part of a successful episode in hindsight
                    else:
                        episode_batch['reward'][ep_idx, t_idx] = [-0.9]

            # max number of transitions to consider for each hindsight episode
            hi_limit = hi_ep_indices_repeat.flatten().shape[0]
            # but at most a quarter of the batch size
            if hi_limit > batch_size // 4:
                hi_limit = batch_size // 4
            # replace a fraction of the batch with hindsight episodes
            episode_idxs = np.concatenate(
                (np.random.choice(episode_idxs, size=batch_size - hi_limit), hi_ep_indices_repeat.flatten()[:hi_limit]))
            # ensure that the concatenated episodes have all of their hindsight transitions inside the batch
            truncated_hi_t_indices = new_hi_t_indices.flatten()[:hi_limit]
            t_samples[-len(truncated_hi_t_indices):] = truncated_hi_t_indices

        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        del transitions['hindsight_instruction']

        return transitions
