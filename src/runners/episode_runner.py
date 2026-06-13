import random
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from adv.attack import attack_gd
from adv.attack_target import attack_target
from components.episode_buffer import EpisodeBatch
from envs import REGISTRY as env_REGISTRY


class EpisodeRunner:
    """Collect one environment episode at a time."""

    GRADIENT_ATTACKS = {"fgsm", "pgd", "PR", "rand_noise", "gaussian"}
    TARGETED_ATTACKS = {"paad", "atsa"}
    RETURN_ADV_ONLY_ATTACKS = {"fgsm", "pgd", "rand_noise", "gaussian"}
    RETURN_ADV_AND_OPP_ATTACKS = {"paad", "atsa", "atla"}

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        self.adv_batch_size = self.args.adv_batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac

    def setup_adv(self, scheme, groups, preprocess, mac, adv_mac):
        self.new_batch = partial(
            EpisodeBatch,
            scheme,
            groups,
            self.batch_size,
            self.episode_limit + 1,
            preprocess=preprocess,
            device=self.args.device,
        )
        self.mac = mac
        self.adv_mac = adv_mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.adv_batch = self.new_batch()
        self.adv_opp_batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def _uses_adv_controller(self):
        return self.args.Number_attack > 0 and self.args.attack_method in self.ADV_CONTROLLER_ATTACKS

    def _build_state_obs_data(self, data, obs):
        return {
            "state": data["state"],
            "avail_actions": data["avail_actions"],
            "obs": obs,
        }

    def _build_post_transition(self, actions, reward, terminated, env_info):
        return {
            "actions": actions,
            "reward": [(reward,)],
            "terminated": [(terminated != env_info.get("episode_limit", False),)],
        }

    def _build_opponent_post_transition(self, clean_obs, perturbations, reward, terminated, env_info):
        return {
            "obs": clean_obs,
            "perturbations": perturbations,
            "reward": [(-reward,)],
            "terminated": [(terminated != env_info.get("episode_limit", False),)],
        }

    def _sample_attacked_agents(self, n_agents):
        return random.sample(range(n_agents), self.args.Number_attack)

    def _apply_agent_subset_perturbations(self, clean_obs, perturbed_obs, attacked_agents):
        attacked_obs = clean_obs.copy()
        for agent_id in attacked_agents:
            attacked_obs[agent_id] = perturbed_obs[agent_id].copy()
        return attacked_obs

    def _sample_atla_perturbations(self, obs_tensor, adv_learner):
        mu, sigma = adv_learner.actor_net(obs_tensor)
        base_mu = torch.squeeze(mu, 0)[0]
        base_sigma = torch.squeeze(sigma, 0)[0]
        return [
            (
                F.hardtanh(torch.distributions.Normal(base_mu, base_sigma).sample())
                * self.args.epsilon_ball
            ).cpu().data.numpy()
            for _ in range(self.args.n_agents)
        ]

    def _collect_atla_step(self, pre_transition_data, hidden_state, adv_learner, obs_shape, n_agents, test_mode):
        clean_obs_tensor = torch.tensor(pre_transition_data["obs"])
        perturbations = self._sample_atla_perturbations(clean_obs_tensor, adv_learner)
        perturbed_obs = np.array(clean_obs_tensor).squeeze(0) + np.array(perturbations)
        attacked_agents = self._sample_attacked_agents(n_agents)
        adv_inputs = self._apply_agent_subset_perturbations(
            pre_transition_data["obs"][0].copy(),
            perturbed_obs,
            attacked_agents,
        )

        adv_transition_data = self._build_state_obs_data(
            pre_transition_data,
            [arr[:obs_shape] for arr in adv_inputs],
        )
        self.adv_batch.update(adv_transition_data, ts=self.t)
        adv_actions, hidden_state_ = self.mac.select_actions(
            self.adv_batch,
            t_ep=self.t,
            t_env=self.t_env,
            hidden_states=hidden_state,
            test_mode=test_mode,
        )
        reward, terminated, env_info = self.env.step(adv_actions[0])
        post_transition_data = self._build_post_transition(adv_actions, reward, terminated, env_info)
        opp_post_transition_data = self._build_opponent_post_transition(
            pre_transition_data["obs"],
            [perturbations],
            reward,
            terminated,
            env_info,
        )

        self.adv_batch.update(post_transition_data, ts=self.t)
        self.adv_opp_batch.update(opp_post_transition_data, ts=self.t)
        self.batch.update(post_transition_data, ts=self.t)
        return reward, terminated, env_info, hidden_state_

    def _collect_clean_step(self, actions, hidden_state_true):
        reward, terminated, env_info = self.env.step(actions[0])
        post_transition_data = self._build_post_transition(actions, reward, terminated, env_info)
        self.batch.update(post_transition_data, ts=self.t)
        return reward, terminated, env_info, hidden_state_true

    def _populate_last_adv_state(self, last_data, actions, learner, adv_learner, obs_shape, n_agents, test_mode, pre_transition_data):
        if self.args.attack_method in self.GRADIENT_ATTACKS:
            adv_inputs = attack_gd(
                self.mac,
                self.batch,
                actions,
                learner.optimiser,
                self.args,
                self.t,
                self.t_env,
                hidden_states=self.hidden_state,
            )
            adv_last_data = self._build_state_obs_data(last_data, [adv_inputs[:, 0:obs_shape]])
            self.adv_batch.update(adv_last_data, ts=self.t)
            adv_actions, _ = self.mac.select_actions(
                self.adv_batch,
                t_ep=self.t,
                t_env=self.t_env,
                hidden_states=self.hidden_state,
                test_mode=test_mode,
            )
            self.adv_batch.update({"actions": adv_actions}, ts=self.t)
            return

        if self.args.attack_method in self.TARGETED_ATTACKS:
            if self.args.attack_method == "atsa":
                tar_actions, _, _ = self.adv_mac.select_actions(
                    self.batch,
                    t_ep=self.t,
                    t_env=self.t_env,
                    hidden_states=self.adv_hidden_state,
                    test_mode=test_mode,
                )
            else:
                tar_actions, _ = self.adv_mac.select_actions(
                    self.batch,
                    t_ep=self.t,
                    t_env=self.t_env,
                    hidden_states=self.adv_hidden_state,
                    test_mode=test_mode,
                )
            adv_inputs = attack_target(
                self.mac,
                self.batch,
                actions,
                tar_actions,
                learner.optimiser,
                self.args,
                self.t,
                self.t_env,
                hidden_state=self.hidden_state,
            )
            adv_last_data = self._build_state_obs_data(last_data, [adv_inputs[:, 0:obs_shape]])
            self.adv_batch.update(adv_last_data, ts=self.t)
            adv_actions, _ = self.mac.select_actions(
                self.adv_batch,
                t_ep=self.t,
                t_env=self.t_env,
                hidden_states=self.hidden_state,
                test_mode=test_mode,
            )
            self.adv_batch.update({"actions": adv_actions}, ts=self.t)
            self.adv_opp_batch.update(last_data, ts=self.t)
            self.adv_opp_batch.update({"actions": tar_actions}, ts=self.t)
            return

        if self.args.attack_method == "atla":
            clean_obs_tensor = torch.tensor(last_data["obs"])
            perturbations = self._sample_atla_perturbations(clean_obs_tensor, adv_learner)
            adv_inputs = np.array(clean_obs_tensor).squeeze(0) + np.array(perturbations)
            adv_last_data = self._build_state_obs_data(last_data, [adv_inputs[:, 0:obs_shape]])
            self.adv_batch.update(adv_last_data, ts=self.t)
            adv_actions, _ = self.mac.select_actions(
                self.adv_batch,
                t_ep=self.t,
                t_env=self.t_env,
                hidden_states=self.hidden_state,
                test_mode=test_mode,
            )
            self.adv_batch.update({"actions": adv_actions}, ts=self.t)
            self.adv_opp_batch.update(last_data, ts=self.t)
            return

    def run(self, test_mode=False, learner=None, adv_test=False, adv_learner=None):
        self.reset()

        terminated = False
        episode_return = 0
        self.hidden_state = self.mac.init_hidden(batch_size=self.batch_size)
        if self._uses_adv_controller():
            self.adv_hidden_state = self.adv_mac.init_hidden(batch_size=self.batch_size)

        env_info = self.env.get_env_info()
        obs_shape = env_info["obs_shape"]
        n_agents = env_info["n_agents"]
        pre_transition_data = None

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)
            actions, hidden_state_true = self.mac.select_actions(
                self.batch,
                t_ep=self.t,
                t_env=self.t_env,
                hidden_states=self.hidden_state,
                test_mode=test_mode,
            )

            if self.args.Number_attack > 0 and adv_test:
                if self.args.attack_method in self.GRADIENT_ATTACKS:
                    adv_inputs = attack_gd(
                        self.mac,
                        self.batch,
                        actions,
                        learner.optimiser,
                        self.args,
                        self.t,
                        self.t_env,
                        self.hidden_state,
                    )
                    adv_transition_data = self._build_state_obs_data(
                        pre_transition_data,
                        [adv_inputs[:, 0:obs_shape]],
                    )
                    self.adv_batch.update(adv_transition_data, ts=self.t)

                    adv_actions, hidden_state_ = self.mac.select_actions(
                        self.adv_batch,
                        t_ep=self.t,
                        t_env=self.t_env,
                        hidden_states=self.hidden_state,
                        test_mode=test_mode,
                    )
                    reward, terminated, env_info = self.env.step(adv_actions[0])
                    post_transition_data = self._build_post_transition(
                        adv_actions, reward, terminated, env_info
                    )
                    self.adv_batch.update(post_transition_data, ts=self.t)
                    self.batch.update(post_transition_data, ts=self.t)
                    self.hidden_state = hidden_state_
                elif self.args.attack_method in self.TARGETED_ATTACKS:
                    if self.args.attack_method == "atsa":
                        tar_actions, adv_hidden_state_, _ = self.adv_mac.select_actions(
                            self.batch,
                            t_ep=self.t,
                            t_env=self.t_env,
                            hidden_states=self.adv_hidden_state,
                            test_mode=test_mode,
                        )
                    else:
                        tar_actions, adv_hidden_state_ = self.adv_mac.select_actions(
                            self.batch,
                            t_ep=self.t,
                            t_env=self.t_env,
                            hidden_states=self.adv_hidden_state,
                            test_mode=test_mode,
                        )
                    adv_inputs = attack_target(
                        self.mac,
                        self.batch,
                        actions,
                        tar_actions,
                        learner.optimiser,
                        self.args,
                        self.t,
                        self.t_env,
                        self.hidden_state,
                    )
                    adv_transition_data = self._build_state_obs_data(
                        pre_transition_data,
                        [adv_inputs[:, 0:obs_shape]],
                    )

                    self.adv_batch.update(adv_transition_data, ts=self.t)
                    self.adv_opp_batch.update(pre_transition_data, ts=self.t)

                    adv_actions, hidden_state_ = self.mac.select_actions(
                        self.adv_batch,
                        t_ep=self.t,
                        t_env=self.t_env,
                        hidden_states=self.hidden_state,
                        test_mode=test_mode,
                    )
                    reward, terminated, env_info = self.env.step(adv_actions[0])
                    post_transition_data = self._build_post_transition(
                        adv_actions, reward, terminated, env_info
                    )
                    opp_post_transition_data = {
                        "actions": tar_actions,
                        "reward": [(-reward,)],
                        "terminated": [(terminated != env_info.get("episode_limit", False),)],
                    }
                    self.adv_batch.update(post_transition_data, ts=self.t)
                    self.adv_opp_batch.update(opp_post_transition_data, ts=self.t)
                    self.batch.update(post_transition_data, ts=self.t)
                    self.hidden_state = hidden_state_
                    self.adv_hidden_state = adv_hidden_state_
                elif self.args.attack_method == "atla":
                    reward, terminated, env_info, hidden_state_ = self._collect_atla_step(
                        pre_transition_data,
                        self.hidden_state,
                        adv_learner,
                        obs_shape,
                        n_agents,
                        test_mode,
                    )
                    self.hidden_state = hidden_state_
                else:
                    reward, terminated, env_info, hidden_state_ = self._collect_clean_step(
                        actions,
                        hidden_state_true,
                    )
                    self.hidden_state = hidden_state_
            else:
                reward, terminated, env_info, hidden_state_ = self._collect_clean_step(
                    actions,
                    hidden_state_true,
                )
                self.hidden_state = hidden_state_

            episode_return += reward
            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
        }
        self.batch.update(last_data, ts=self.t)
        actions, _ = self.mac.select_actions(
            self.batch,
            t_ep=self.t,
            t_env=self.t_env,
            hidden_states=self.hidden_state,
            test_mode=test_mode,
        )
        self.batch.update({"actions": actions}, ts=self.t)

        if self.args.Number_attack > 0 and adv_test:
            self._populate_last_adv_state(
                last_data,
                actions,
                learner,
                adv_learner,
                obs_shape,
                n_agents,
                test_mode,
                pre_transition_data,
            )

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""

        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)
        if self.args.evaluate:
            print(episode_return, "-------------", cur_stats["battle_won"])
        if test_mode and (len(self.test_returns) == self.args.test_nepisode - 1):
            self._log(cur_returns, cur_stats, log_prefix)

        if self.args.Number_attack > 0 and adv_test:
            if self.args.attack_method in self.RETURN_ADV_ONLY_ATTACKS:
                return self.adv_batch
            if self.args.attack_method == "PR":
                return self.batch, self.adv_batch
            if self.args.attack_method in self.RETURN_ADV_AND_OPP_ATTACKS:
                return self.adv_batch, self.adv_opp_batch
        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)

        returns.clear()
        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)

        stats.clear()
