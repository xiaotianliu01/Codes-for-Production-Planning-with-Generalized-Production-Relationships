import numpy as np
import pandas as pd

class Production_Net():

    def __init__(self, seed, data_pth, instance_id, lam):

        np.random.seed(seed)
        data = pd.read_excel(data_pth,sheet_name=None)
        top_data = data[str(instance_id) + '_LL']
        node_data = data[str(instance_id) + '_SD']
        self.num_product = node_data['Stage Name'].shape[0]
        
        self.name_to_index = {}
        self.level_product_set = {}
        node_names = node_data['Stage Name']
        node_depth = node_data['relDepth']
        level = 0
        cnt = 0
        while(True):
            flag = False
            for i in range(self.num_product):
                if(int(node_depth[i]) == level):
                    self.name_to_index[node_names[i]] = cnt
                    cnt += 1
                    if(level in self.level_product_set.keys()):
                        self.level_product_set[level].append(node_names[i])
                    else:
                        self.level_product_set[level] = [node_names[i]]
                    flag = True
            if(flag == False):
                break
            level += 1
        self.num_level = level

        self.index_level_set = {}
        for level in self.level_product_set.keys():
            for product_name in self.level_product_set[level]:
                self.index_level_set[self.name_to_index[product_name]] = level
                
        self.num_final_product = len(self.level_product_set[0])
        self.M = np.zeros([self.num_product, self.num_product])
        for i in range(top_data.shape[0]):
            source_node = top_data['sourceStage'][i]
            destination_node = top_data['destinationStage'][i]
            self.M[self.name_to_index[source_node]][self.name_to_index[destination_node]] = 1
        
        self.demand_mean = [0 for i in range(self.num_final_product)]
        self.demand_std = [0 for i in range(self.num_final_product)]
        service_level = [0 for i in range(self.num_final_product)]

        for j in range(self.num_product):
            if(node_names[j] in self.level_product_set[0]):
                self.demand_mean[self.name_to_index[node_names[j]]] = float(node_data['avgDemand'][j])/10 # need to change to smaller values
                self.demand_std[self.name_to_index[node_names[j]]] = float(node_data['stdDevDemand'][j])/10
                service_level[self.name_to_index[node_names[j]]] = float(node_data['serviceLevel'][j])

        service_level = service_level + [0.99 for _ in range(self.num_product - self.num_final_product)]
        self.demand_mean = self.demand_mean + [0]*(self.num_product - self.num_final_product)
        self.demand_std = self.demand_std + [0]*(self.num_product - self.num_final_product)
        
        for i in range(self.num_final_product, self.num_product):
            ratio = self.get_to_final_product_ratio(i)
            t_m = 0
            t_v = 0
            for j in range(self.num_final_product):
                t_m += ratio[j]*self.demand_mean[j]
                t_v += (ratio[j]*self.demand_std[j])**2
            self.demand_mean[i] = t_m
            self.demand_std[i] = t_v**0.5
        
        self.hs = [1]*self.num_product
        self.ss = [self.demand_mean[i] for i in range(self.num_product)]
        self.bs = [1/(1/service_level[i]-1) for i in range(self.num_product)]
        self.es = [2*self.hs[i]*self.demand_mean[i] for i in range(self.num_product)]
        
        self.demand_state_dim = 0
        self.operations_num = 3
        self.prod_state_dim = [5*self.operations_num for i in range(self.num_product)]
        self.process_time_mean = [np.array([0.95/self.demand_mean[i] for j in range(self.operations_num)]) for i in range(self.num_product)]
        self.initial_inventory = [int(self.demand_mean[i]) for i in range(self.num_product)]
        self.initial_backlog = [0 for i in range(self.num_product)]
        self.lam = lam
    
    def get_to_final_product_ratio(self, i):
        res = np.zeros([self.num_final_product])
        if(i<self.num_final_product):
            res[i] = 1
        else:
            for j in range(self.num_product):
                if(self.M[i][j] > 0):
                    res += self.M[i][j]*self.get_to_final_product_ratio(j)
        return res

    def init_prod_state(self, thread_num):

        self.jobs_set = [[[[] for j in range(self.operations_num)] for i in range(self.num_product)] for k in range(thread_num)]
        prod_states = []
        for i in range(self.num_product):
            t = np.array([0]*self.prod_state_dim[i])
            prod_states.append(t)
        
        thread_prod_states = []
        for i in range(self.num_product):
            thread_prod_states.append(np.concatenate([np.array([prod_states[i]])]*thread_num))
        
        return thread_prod_states
    
    def init_demand_state(self, thread_num):
        return None

    def job_ordering(self, sched, job_list, operations_index):
        policy_index = np.random.choice([i for i in range(5)], 1, p=sched)
        if(policy_index == 0): #SPT
            return sorted(job_list, key=lambda x: x[operations_index])
        if(policy_index == 1): #LPT
            return sorted(job_list, key=lambda x: -x[operations_index])
        elif(policy_index == 2): #LWKR
            return sorted(job_list, key=lambda x: np.sum(x[operations_index:-1]))
        elif(policy_index == 3): #MWKR
            return sorted(job_list, key=lambda x: -np.mean(x[operations_index:-1]))
        elif(policy_index == 4): #AT-RPT
            return sorted(job_list, key=lambda x: -x[-1]-np.sum(x[operations_index:-1]))

    def simulate_flow_shop(self, job_list):
        t = 0
        production = 0
        while(True):
            t_list = []
            for i in range(self.operations_num):
                if(len(job_list[i]) != 0):
                    t_list.append(job_list[i][0][i])
                else:
                    t_list.append(np.inf)

            min_t = np.min(t_list)
            if(min_t == np.inf):
                return production, job_list
            if(t + min_t > 1):
                min_t = 1 - t
                for i in range(self.operations_num):
                    if(len(job_list[i]) != 0):
                        job_list[i][0][i] -= min_t
                        for k in range(len(job_list[i])):
                            job_list[i][k][-1] += min_t
                return production, job_list
            
            finished = np.argmin(t_list)
            for i in range(self.operations_num):
                if(len(job_list[i]) != 0):
                    job_list[i][0][i] -= min_t
                    for k in range(len(job_list[i])):
                        job_list[i][k][-1] += min_t

            if(finished == self.operations_num - 1):
                production += 1
            else:
                job_list[finished+1].append(job_list[finished][0])
            job_list[finished] = job_list[finished][1:]
            t += min_t

    def update_prod_state(self, old_s, Qs, sched, step_num, thread_num):
        s = [[] for _ in range(self.num_product)]
        productions = np.ones([thread_num, self.num_product])
        scheduling_costs = np.zeros([thread_num, self.num_product])
        
        for k in range(thread_num):
            for i in range(self.num_product):
                new_jobs = np.random.exponential(self.process_time_mean[i], (int(Qs[k,i]), self.operations_num))
                new_jobs = np.concatenate([new_jobs, np.zeros([int(Qs[k,i]), 1,])], axis=1)
                for l in range(new_jobs.shape[0]):
                    self.jobs_set[k][i][0].append(new_jobs[l,])
                for operations_index in range(self.operations_num):
                    self.jobs_set[k][i][operations_index] = self.job_ordering(sched[k,i,:], self.jobs_set[k][i][operations_index], operations_index)

                production, self.jobs_set[k][i] = self.simulate_flow_shop(self.jobs_set[k][i])
                productions[k][i] = production
                
                ls = [0]*self.operations_num
                for operations_index in range(self.operations_num):
                    if(len(self.jobs_set[k][i][operations_index]) > 0):
                        ls[operations_index] = max(ls[operations_index], np.max([job[-1] for job in self.jobs_set[k][i][operations_index]]))
                scheduling_costs[k][i] = np.max(ls)*self.ss[i]
                
                t_s = []
                for operations_index in range(self.operations_num):
                    if(len(self.jobs_set[k][i][operations_index]) == 0):
                        t_s += [0, 0, 0, 0, 0]
                    else:
                        t_s.append(len(self.jobs_set[k][i][operations_index]))
                        t_s.append(np.min(np.stack(self.jobs_set[k][i][operations_index], axis = 0)[:,operations_index]))
                        t_s.append(np.max(np.stack(self.jobs_set[k][i][operations_index], axis = 0)[:,operations_index]))
                        t_s.append(np.sum(np.stack(self.jobs_set[k][i][operations_index], axis = 0)[:,operations_index]))
                        t_s.append(ls[operations_index])
                s[i].append(np.array(t_s))

        s = [np.array(s_t) for s_t in s]
        return s, productions, scheduling_costs
    
    def update_demand_state(self, old_s, step_num, thread_num):
        s = []
        for i in range(self.num_final_product):
            d = np.random.gamma(self.demand_mean[i]**2/(self.demand_std[i]**2), (self.demand_std[i])**2/self.demand_mean[i], thread_num)
            s.append(d)
        s = np.stack(s, axis = 1)
        s = (s>0)*s
        s = s.astype(int)
        return None, s
    
    def cost_2_rewards(self, costs):
        return -(1-self.lam)*costs - self.lam*np.reshape(np.sum(costs, axis=1), [costs.shape[0], 1])
        