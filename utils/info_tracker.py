import numpy as np
from copy import deepcopy as dc

class infos_tracker():
    
    def __init__(self, n_rollout_threads, num_agents):
        
        self.rewards_log = []
        self.inv_log = []
        self.backlog_log = []
        self.Q_log = []
        self.demand_log = []
        self.holding_cost_log = []
        self.backlog_cost_log = []
        self.set_up_cost_log = []
        self.n_rollout_threads = n_rollout_threads
        self.num_agents = num_agents

    def insert_info(self, infos, rewards):
        self.inv_log.append(infos[0]['inventory'])
        self.backlog_log.append(infos[0]['backlog'])
        self.Q_log.append(infos[0]['Q'])
        self.demand_log.append(infos[0]['demands'])
        self.holding_cost_log.append(infos[0]['holding costs'])
        self.backlog_cost_log.append(infos[0]['backlog costs'])
        self.set_up_cost_log.append(infos[0]['set up costs'])
        self.rewards_log.append(rewards.squeeze(-1))
    
    def compute_average(self):
        
        def get_average(data):
            t = np.stack(data, axis = 0)
            t = np.mean(t, axis = 0)
            return t
        
        self.inv_log = get_average(self.inv_log)
        self.backlog_log = get_average(self.backlog_log)
        self.Q_log = get_average(self.Q_log)
        self.demand_log = get_average(self.demand_log)
        self.holding_cost_log = get_average(self.holding_cost_log)
        self.backlog_cost_log = get_average(self.backlog_cost_log)
        self.set_up_cost_log = get_average(self.set_up_cost_log)
        self.rewards_log = get_average(self.rewards_log)

    def print_info_summary(self):
        rew = [round(np.mean(self.rewards_log, axis=0)[l], 2) for l in range(self.num_agents)]
        inv = [round(np.mean(self.inv_log, axis=0)[l], 2) for l in range(self.num_agents)]
        bac = [round(np.mean(self.backlog_log, axis=0)[l], 2) for l in range(self.backlog_log.shape[1])]
        inv_p = dc(inv)
        for i in range(len(bac)):
            inv_p[i] = inv_p[i] - bac[i]
        act = [round(np.mean(self.Q_log, axis=0)[l], 2) for l in range(self.num_agents)]
        dem = [round(np.mean(self.demand_log, axis=0)[l], 2) for l in range(self.num_agents)]
        print("Reward: " + str(rew) + " " + str(round(np.mean(rew),2)) + "  Inventory: " + str(inv_p) +"  Order: " + str(act) + " Demand: " + str(dem))
    
    def clear(self):
        
        self.rewards_log = []
        self.inv_log = []
        self.backlog_log = []
        self.Q_log = []
        self.demand_log = []
        self.holding_cost_log = []
        self.backlog_cost_log = []
        self.set_up_cost_log = []