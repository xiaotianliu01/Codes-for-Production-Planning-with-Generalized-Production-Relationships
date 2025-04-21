"""
# @Time    : 2021/7/1 7:14 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_runner.py
"""

import time
import os
import numpy as np
from itertools import chain
from copy import deepcopy as dc
import torch
from utils.util import update_linear_schedule
from utils.info_tracker import infos_tracker
from runner.separated.base_runner import Runner
import matplotlib as mpl
import matplotlib.pyplot as plt

def _t2n(x):
    return x.detach().cpu().numpy()

class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    def run(self):

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        best_cost = np.inf
        record = 0
        tracker = infos_tracker(self.n_rollout_threads, self.num_agents)

        for episode in range(episodes):
            
            if episode % self.eval_interval == 0 and self.use_eval:
                re = self.eval(episode)
                if(self.all_args.verbose == True):
                    print("evaluation cost: ", re, " current best cost: ", best_cost)
                if(re < best_cost):
                    self.save()
                    best_cost = re
                    record = 0
                else:
                    record += 1
                    if(record == self.all_args.no_improvement_episodes):
                        print("Training Finished !", best_cost)
                        return
            
            self.warmup()
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                
                tracker.insert_info(infos, rewards)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)
            # compute return and update network
            self.compute()
            train_infos = self.train(int(episode/self.all_args.update_per_agent)%self.num_agents)
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                if(self.all_args.verbose == True):
                    print("\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))
                    tracker.compute_average()
                    tracker.print_info_summary()
                tracker.clear()
            # eval

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                action_env = action

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))
        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                             dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                    dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(share_obs,
                                         np.array(list(obs[:, agent_id])),
                                         rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id],
                                         actions[:, agent_id],
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id],
                                         rewards[:, agent_id],
                                         masks[:, agent_id])

    @torch.no_grad()
    def eval(self, episode):
        overall_reward = []
        eval_train_infos = [[] for _ in range(self.num_agents)]
        for _ in range(self.all_args.eval_episodes):
            eval_episode_rewards = []
            eval_obs = self.eval_envs.reset()

            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                    dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for eval_step in range(self.episode_length):
                eval_temp_actions_env = []
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].prep_rollout()
                    eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                    eval_rnn_states[:, agent_id],
                                                                                    eval_masks[:, agent_id],
                                                                                    deterministic=True)
                    eval_action = eval_action.detach().cpu().numpy()
                    # rearrange action
                    if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.eval_envs.action_space[agent_id].shape):
                            eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i] + 1)[
                                eval_action[:, i]]
                            if i == 0:
                                eval_action_env = eval_uc_action_env
                            else:
                                eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                    elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                    else:
                        eval_action_env = eval_action

                    eval_temp_actions_env.append(eval_action_env)
                    eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

                # [envs, agents, dim]
                eval_actions_env = []
                for i in range(self.n_eval_rollout_threads):
                    eval_one_hot_action_env = []
                    for eval_temp_action_env in eval_temp_actions_env:
                        eval_one_hot_action_env.append(eval_temp_action_env[i])
                    eval_actions_env.append(eval_one_hot_action_env)

                # Obser reward and next obs
                eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
                eval_episode_rewards.append(eval_rewards)
                overall_reward.append(np.mean(eval_rewards))

                eval_rnn_states[eval_dones == True] = np.zeros(
                    ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

            eval_episode_rewards = np.array(eval_episode_rewards)

            for agent_id in range(self.num_agents):
                eval_average_episode_rewards = np.mean(eval_episode_rewards[:, :, agent_id, :])
                eval_train_infos[agent_id].append(eval_average_episode_rewards)
        
        eval_train_infos = [np.mean(i) for i in eval_train_infos]
        eval_train_infos.append(np.mean(overall_reward))
        self.log_train(eval_train_infos, episode)
        return np.mean(overall_reward)