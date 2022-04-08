import numpy as np


class RolloutWorker:

    def __init__(self, env, policy, cfg, env_params, language_conditioned=False):

        self.env = env
        self.policy = policy
        self.cfg = cfg
        self.env_params = env_params
        self.language_conditioned = language_conditioned
        self.timestep_counter = 0

    def generate_rollout(self, train_mode=False, animated=False):

        episodes = []
        for _ in range(self.cfg.num_rollouts_per_mpi):
            ep_obs, ep_actions, ep_success, ep_rewards = [], [], [], []
            observation = self.env.reset()
            obs = observation['observation']

            if self.language_conditioned:
                instruction = observation['instruction']
                ep_instructions, ep_hinsight_instruction = [], []
            else:
                ag = observation['achieved_goal']
                g = observation['desired_goal']
                ep_ag, ep_g = [], []

            for _ in range(self.env_params['max_timesteps']):
                if self.language_conditioned:
                    action = self.policy.act(obs.copy(), instruction.copy(), train_mode)
                else:
                    action = self.policy.act(obs.copy(), ag.copy(), g.copy(), train_mode)
                if animated:
                    self.env.render()

                observation_new, reward, _, info = self.env.step(action)
                self.timestep_counter += 1

                obs_new = observation_new['observation']

                if self.language_conditioned:
                    instruction_new = observation_new['instruction']
                    hindsight_instr = info['hindsight_instruction'] if 'hindsight_instruction' in info.keys(
                    ) else np.zeros_like(instruction_new)
                else:
                    ag_new = observation_new['achieved_goal']

                ep_obs.append(obs.copy())
                ep_actions.append(action.copy())
                ep_rewards.append([reward])
                if self.language_conditioned:
                    ep_instructions.append(instruction.copy())
                    ep_hinsight_instruction.append(hindsight_instr.copy())
                else:
                    ep_ag.append(ag.copy())
                    ep_g.append(g.copy())

                obs = obs_new
                if self.language_conditioned:
                    instruction = instruction_new
                else:
                    ag = ag_new
                ep_success.append(info['is_success'])

            ep_obs.append(obs.copy())
            if not self.language_conditioned:
                ep_ag.append(ag.copy())

            episode_data = dict(obs=np.array(ep_obs).copy(),
                                action=np.array(ep_actions).copy(),
                                reward=np.array(ep_rewards).copy(),
                                success=np.array(ep_success).copy(),
                                timesteps=self.timestep_counter)

            if self.language_conditioned:
                episode_data['instruction'] = np.array(ep_instructions).copy()
                episode_data['hindsight_instruction'] = np.array(ep_hinsight_instruction).copy()
            else:
                episode_data['g'] = np.array(ep_g).copy()
                episode_data['ag'] = np.array(ep_ag).copy()

            episodes.append(episode_data)

        return episodes

    def generate_test_rollout(self, animated=False):
        rollout_data = []
        for _ in range(self.cfg.n_test_rollouts):
            rollout = self.generate_rollout(train_mode=False, animated=animated)
            rollout_data.append(rollout)
        # only take the last step to calculate success rate
        success_rate = np.mean([_rd['success'][-1] for rd in rollout_data for _rd in rd])
        rewards = np.sum([_rd['reward'] for rd in rollout_data for _rd in rd], 1).mean()
        return success_rate, rewards
