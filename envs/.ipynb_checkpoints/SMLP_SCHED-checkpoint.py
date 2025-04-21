import numpy as np
import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

class Env(object):

    def __init__(self, seed, data_pth, instance_id, lam, episode_max_step, thread_num=1, exp_name='Fixed_Production_Lead_Time'):

        self.seed(seed)
        self.thread_num = thread_num
        self.episode_max_steps = episode_max_step
        
        if(exp_name == 'One_Stage_Parrellel'):
            from .Prod_Net_One_Stage_Parrellel import Production_Net
            self.prod_net = Production_Net(seed, data_pth, instance_id, lam)

        self.agent_num = self.prod_net.num_product
        self.final_product_num = self.prod_net.num_final_product
        self.M = self.prod_net.M
        self.hs = self.prod_net.hs
        self.bs = self.prod_net.bs 
        self.es = self.prod_net.es 
        prod_state_dim = self.prod_net.prod_state_dim
        demand_state_dim = self.prod_net.demand_state_dim
        self.norm_mean = self.prod_net.demand_mean 
        self.norm_std = self.prod_net.demand_std
        self.initial_inventory = [np.array(self.prod_net.initial_inventory) for _ in range(self.thread_num)]
        self.initial_backlog = [np.array(self.prod_net.initial_backlog) for _ in range(self.thread_num)]
        self.obs_dim = self.agent_num + np.sum(prod_state_dim) + np.sum(demand_state_dim)
        self.action_dim = 1 + prod_state_dim[0]
    
    def reset(self, train = True):

        self.eval_log = [] ###
        self.eval_log_action = [] ###

        self.train = train
        self.inventory = np.array(self.initial_inventory)
        self.backlog = np.array(self.initial_backlog)
        self.lost_sales = np.zeros([self.thread_num, self.agent_num-self.final_product_num])
        self.step_num = 0

        self.prod_states = self.prod_net.init_prod_state(self.thread_num)
        self.demand_states = self.prod_net.init_demand_state(self.thread_num)
        return self.assembly_state()
    
    def seed(self, seed=None):
        np.random.seed(seed)

    def denormalize(self, x):
        return x*self.norm_std + self.norm_mean
    
    def softmax(self, x):
        exp_values = np.exp(x)
        sum = np.sum(exp_values, axis=2).reshape([exp_values.shape[0], exp_values.shape[1], 1])
        return exp_values/sum

    def assembly_state(self):
        sub_agent_obs = [self.inventory[:,i] - self.backlog[:,i] for i in range(self.final_product_num)]
        sub_agent_obs += [self.inventory[:,i] for i in range(self.final_product_num, self.agent_num)]
        sub_agent_obs = np.stack(sub_agent_obs, axis = 1)

        if(self.prod_states):
            concat_prod_states = np.concatenate(self.prod_states, axis=1)
            sub_agent_obs = np.concatenate([sub_agent_obs, concat_prod_states], axis=1)
        if(self.demand_states):
            concat_demand_states = np.concatenate(self.demand_states, axis=1)
            sub_agent_obs = np.concatenate([sub_agent_obs, concat_demand_states], axis=1)
        agent_obs = [sub_agent_obs for i in range(self.agent_num)]
        return agent_obs
    
    def step(self, actions, denormalize = True):

        sched_actions = self.softmax(actions[:,:,1:])
        actions = actions[:,:,0].squeeze()
        if(denormalize):
            action = self.denormalize(actions)
            action = (action > 0)*action 
            action = action.astype(int)
        else:
            action = np.array(actions)

        costs, eval_costs, self.prod_states, self.demand_states, infos = self.state_update(action, sched_actions)
        '''
        if(self.train==False):###
            t = []
            tt = []
            cnt = 0
            for prod_state in self.prod_states:
                t.append(np.mean(prod_state[:,1]))
                tt.append(np.mean(sched_actions[:,cnt,0]))
                cnt += 1
            self.eval_log.append(np.array(t))####
            self.eval_log_action.append(np.array(tt))
        if(self.step_num == 100):
            print(np.mean(np.array(self.eval_log), axis=0))
            print(np.mean(np.array(self.eval_log_action)))
        '''
        state = self.assembly_state()

        if(self.train):
            rewards = self.prod_net.cost_2_rewards(costs)
        else:
            rewards = eval_costs

        if(self.step_num > self.episode_max_steps):
            sub_agent_done = [np.ones([self.thread_num]) for i in range(self.agent_num)]
        else:
            sub_agent_done = [np.zeros([self.thread_num]) for i in range(self.agent_num)]
        sub_agent_info = [infos for i in range(self.agent_num)]

        return [state, rewards, sub_agent_done, sub_agent_info]        
    
    def state_update(self, action, sched_actions):
        
        infos = {'demands': None, 'Q': None, 'inventory':None, 'backlog':None, 'lost sales':None, 'holding costs':[], 'backlog costs':[], 'set up costs':[]}
        
        new_demand_states, demand = self.prod_net.update_demand_state(self.demand_states, self.step_num, self.thread_num)
        for i in range(self.final_product_num, self.agent_num):
            demand = np.concatenate([demand, np.reshape(np.sum(self.M[i,:]*action, axis=1), [self.thread_num, 1])], axis=1)
        infos['demands'] = demand

        Q = []
        for i in range(self.agent_num):
            temp = [np.ones(self.thread_num)]
            for j in range(self.agent_num):
                if(self.M[j, i] > 0):
                    temp.append(self.inventory[:,j]/(demand[:,j]+1e-6))
            temp = np.stack(temp, axis = 1)
            q = action[:,i]*np.min(temp, axis=1)
            Q.append(q.astype(int))
        Q = np.stack(Q, axis=1)
        infos['Q'] = Q

        new_prod_states, production = self.prod_net.update_prod_state(self.prod_states, Q, sched_actions, self.step_num, self.thread_num)

        new_inventory = []
        new_backlog = []
        for i in range(self.final_product_num):
            po = self.inventory[:,i] - self.backlog[:,i] - demand[:,i]
            new_inventory.append((po > 0)*po + production[:, i])
            new_backlog.append((-po > 0)*(-po))

        lost_sales = []
        for i in range(self.final_product_num, self.agent_num):
            po = demand[:,i]-self.inventory[:,i]
            lost_sales.append((po>0)*po)
            inv = self.inventory[:,i] + production[:,i] - np.sum(self.M[i,:]*Q, axis=1)
            new_inventory.append(inv)

        self.lost_sales = np.stack(lost_sales, axis=1)
        self.inventory = np.stack(new_inventory, axis=1)
        self.backlog = np.stack(new_backlog, axis=1)
        infos['inventory'] = self.inventory
        infos['backlog'] = self.backlog
        infos['lost sales'] = self.lost_sales

        costs = []
        eval_costs = []
        for i in range(self.final_product_num):
            costs.append(self.hs[i]*self.inventory[:,i] + self.bs[i]*self.backlog[:,i] + self.es[i]*(Q[:,i]>0))
            eval_costs.append(self.hs[i]*self.inventory[:,i] + self.bs[i]*self.backlog[:,i] + self.es[i]*(Q[:,i]>0))
            infos['holding costs'].append(self.hs[i]*self.inventory[:,i])
            infos['backlog costs'].append(self.bs[i]*self.backlog[:,i])
            infos['set up costs'].append(self.es[i]*(Q[:,i]>0))
        for i in range(self.final_product_num, self.agent_num):
            costs.append(self.hs[i]*self.inventory[:,i] + self.bs[i]*self.lost_sales[:,i-self.final_product_num] + self.es[i]*(Q[:,i]>0))
            eval_costs.append(self.hs[i]*self.inventory[:,i] + self.es[i]*(Q[:,i]>0))
            infos['holding costs'].append(self.hs[i]*self.inventory[:,i])
            infos['set up costs'].append(self.es[i]*(Q[:,i]>0))
            
        costs = np.stack(costs, axis=1)
        eval_costs = np.stack(eval_costs, axis=1)
        infos['holding costs'] = np.stack(infos['holding costs'], axis=1)
        infos['backlog costs'] = np.stack(infos['backlog costs'], axis=1)
        infos['set up costs'] = np.stack(infos['set up costs'], axis=1)
        self.step_num += 1
        
        return costs, eval_costs, new_prod_states, new_demand_states, infos
