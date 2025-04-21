# !/usr/bin/env python
import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from config import get_config
from envs.env_wrappers import SubprocVecEnv
from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy
from utils.info_tracker import infos_tracker
import pandas as pd

def _t2n(x):
    return x.detach().cpu().numpy()
    
def eval(all_args, eval_envs, device):

    policy = []
    for agent_id in range(all_args.num_agents):
        share_observation_space = eval_envs.share_observation_space[agent_id] if all_args.use_centralized_V else eval_envs.observation_space[agent_id]

        po = Policy(all_args,
                    eval_envs.observation_space[agent_id],
                    share_observation_space,
                    eval_envs.action_space[agent_id],
                    device = device)
        policy.append(po)
    
    for agent_id in range(all_args.num_agents):
        policy_actor_state_dict = torch.load(str(all_args.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
        policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
        policy[agent_id].actor.eval()

    tracker = infos_tracker(all_args.n_eval_rollout_threads, all_args.num_agents)    

    eval_obs = eval_envs.reset()
    eval_rnn_states = np.zeros((all_args.n_eval_rollout_threads, all_args.num_agents, all_args.recurrent_N, all_args.hidden_size),
                            dtype=np.float32)
    eval_masks = np.ones((all_args.n_eval_rollout_threads, all_args.num_agents, 1), dtype=np.float32)
    for eval_step in range(all_args.episode_length):
        eval_temp_actions_env = []
        for agent_id in range(all_args.num_agents):
            eval_action, eval_rnn_state = policy[agent_id].act(np.array(list(eval_obs[:, agent_id])),
                                                                            eval_rnn_states[:, agent_id],
                                                                            eval_masks[:, agent_id],
                                                                            deterministic=True)
            eval_action = eval_action.detach().cpu().numpy()
            # rearrange action
            eval_action_env = eval_action
            eval_temp_actions_env.append(eval_action_env)
            eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
        # [envs, agents, dim]
        eval_actions_env = []
        for i in range(all_args.n_eval_rollout_threads):
            eval_one_hot_action_env = []
            for eval_temp_action_env in eval_temp_actions_env:
                eval_one_hot_action_env.append(eval_temp_action_env[i])
            eval_actions_env.append(eval_one_hot_action_env)
        # Obser reward and next obs
        eval_obs, eval_rewards, eval_dones, eval_infos = eval_envs.step(eval_actions_env)
        tracker.insert_info(eval_infos, eval_rewards)
        eval_rnn_states[eval_dones == True] = np.zeros(
            ((eval_dones == True).sum(), all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
        eval_masks = np.ones((all_args.n_eval_rollout_threads, all_args.num_agents, 1), dtype=np.float32)
        eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
    
    tracker.compute_average()
    print('Results for ' + all_args.model_dir.split('/')[-3])
    print('overall costs ', 'mean: ', np.mean(np.sum(tracker.rewards_log, axis=1)), ' std: ', np.std(np.sum(tracker.rewards_log, axis=1)))
    print('holding costs ', 'mean: ', np.mean(np.sum(tracker.holding_cost_log, axis=1)), ' std: ', np.std(np.sum(tracker.holding_cost_log, axis=1)))
    print('backlog costs ', 'mean: ', np.mean(np.sum(tracker.backlog_cost_log, axis=1)), ' std: ', np.std(np.sum(tracker.backlog_cost_log, axis=1)))
    print('set up costs ', 'mean: ', np.mean(np.sum(tracker.set_up_cost_log, axis=1)), ' std: ', np.std(np.sum(tracker.set_up_cost_log, axis=1)))
    
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + '/' + os.path.split(os.path.dirname(os.path.abspath(__file__)))[1] +"/results/") / all_args.env_name / all_args.experiment_name / "evaluation" / all_args.instance_id
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    f_pth = str(run_dir) + '/' + str(all_args.model_dir.split('/')[-3]) + ".csv"
    means = np.array([np.mean(np.sum(tracker.rewards_log, axis=1)), np.mean(np.sum(tracker.holding_cost_log, axis=1)), np.mean(np.sum(tracker.backlog_cost_log, axis=1)), np.mean(np.sum(tracker.set_up_cost_log, axis=1))])
    stds = np.array([np.std(np.mean(tracker.rewards_log, axis=1)), np.std(np.mean(tracker.holding_cost_log, axis=1)), np.std(np.sum(tracker.backlog_cost_log, axis=1)), np.std(np.sum(tracker.set_up_cost_log, axis=1))])
    d = np.stack([means, stds], axis=1)
    d = pd.DataFrame(d, columns = ['mean', 'std'], index = ['cost', 'holding cost', 'backlog cost', 'set up cost'])
    d.to_csv(f_pth)

def make_eval_env(all_args):
    return SubprocVecEnv(all_args, train=False)

def parse_args(args, parser):
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int, default=0, help="number of players")
    all_args = parser.parse_known_args(args)[0]
    return all_args

def main(args):

    parser = get_config()
    all_args = parse_args(args, parser)
    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)
    
    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
                              str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    all_args.num_agents = eval_envs.num_agent
    num_agents = all_args.num_agents

    # run experiments
    eval(all_args, eval_envs, device)

if __name__ == "__main__":
    main(sys.argv[1:])
