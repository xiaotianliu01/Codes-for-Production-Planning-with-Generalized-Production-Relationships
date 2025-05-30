"""
# @Time    : 2021/7/1 8:44 上午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_wrappers.py
Modified from OpenAI Baselines code to work with multi-agent envs
"""

import numpy as np
import gym
from gym import spaces


class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """

    def __init__(self, array_of_param_array):
        super().__init__()
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        # For each row: round(random .* (max - min) + min, 0)
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]

    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (
                    np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


class SubprocVecEnv(object):
    def __init__(self, all_args, train):
        """
        envs: list of gym environments to run in subprocesses
        """
        Env = getattr(getattr(__import__('envs.' + all_args.env_name), all_args.env_name), 'Env')
        self.num_envs = all_args.n_rollout_threads if train else all_args.n_eval_rollout_threads
        self.env_list = Env(all_args.seed, all_args.data_pth, all_args.instance_id, all_args.reward_lambda, all_args.episode_length + 1, self.num_envs, all_args.experiment_name)
        self.train = train
        self.num_agent = self.env_list.agent_num
        self.signal_obs_dim = self.env_list.obs_dim
        self.signal_action_dim = self.env_list.action_dim

        self.u_range = np.finfo(np.float32).max  # control range for continuous control
        self.movable = True

        # environment parameters
        # self.discrete_action_space = True
        self.discrete_action_space = False

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = False

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in range(self.num_agent):
            total_action_space = []

            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(self.signal_action_dim)
            else:
                u_action_space = spaces.Box(np.array([-self.u_range]*self.signal_action_dim), np.array([self.u_range]*self.signal_action_dim), dtype=np.float32)  # [-1,1]
            if self.movable:
                total_action_space.append(u_action_space)

            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # observation space
            share_obs_dim += self.signal_obs_dim
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_dim,),
                                                     dtype=np.float32))  # [-inf,inf]

        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,),
                                                   dtype=np.float32) for _ in range(self.num_agent)]

    def step(self, actions):
        
        action = np.zeros([len(actions), len(actions[0]), self.signal_action_dim])
        for i in range(len(actions)):
            for j in range(len(actions[0])):
                for k in range(self.signal_action_dim):
                    action[i][j][k] = actions[i][j][k]
        obs, rews, dones, infos = self.env_list.step(action)
        return np.stack(obs, axis=1), np.reshape(np.stack(rews), [np.stack(rews).shape[0], np.stack(rews).shape[1], 1]), np.stack(dones, axis=1), infos
         
    def reset(self):
        obs = self.env_list.reset(train=self.train)
        return np.stack(obs, axis=1)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass
