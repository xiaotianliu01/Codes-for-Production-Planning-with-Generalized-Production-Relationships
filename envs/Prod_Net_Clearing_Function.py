import numpy as np
import pandas as pd
import math

class Production_Net():

    def __init__(self, seed, data_pth, instance_id, lam, func_form = 'frac'): # frac, exp, poly 

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
        self.num_final_product = len(self.level_product_set[0])
        self.M = np.zeros([self.num_product, self.num_product])
        for i in range(top_data.shape[0]):
            source_node = top_data['sourceStage'][i]
            destination_node = top_data['destinationStage'][i]
            self.M[self.name_to_index[source_node]][self.name_to_index[destination_node]] = 1
        
        self.demand_mean = [0 for i in range(self.num_final_product)]
        self.demand_std = [0 for i in range(self.num_final_product)]
        service_level = [0 for i in range(self.num_final_product)]
        self.processing_time_mean = [0 for i in range(self.num_product)]

        for j in range(self.num_product):
            if(node_names[j] in self.level_product_set[0]):
                self.demand_mean[self.name_to_index[node_names[j]]] = float(node_data['avgDemand'][j])
                self.demand_std[self.name_to_index[node_names[j]]] = float(node_data['stdDevDemand'][j])
                service_level[self.name_to_index[node_names[j]]] = float(node_data['serviceLevel'][j])
            self.processing_time_mean[self.name_to_index[node_names[j]]] = int(node_data['stageTime'][j]/10) + 1

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
        self.clear_functions = [self.get_clearing_func(func_form, i) for i in range(self.num_product)]
        self.hs = [1]*self.num_product
        self.bs = [1/(1/service_level[i]-1) for i in range(self.num_product)]
        self.es = [2*self.hs[i]*self.demand_mean[i] for i in range(self.num_product)]
        
        self.demand_state_dim = 0
        self.prod_state_dim = [1]*self.num_product
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
    
    def get_clearing_func(self, func_form, i):

        if(func_form == 'frac'):
            K_1 = self.demand_mean[i]*3
            K_2 = self.processing_time_mean[i]*K_1 - self.demand_mean[i]
            return lambda x: min(x, K_1*x/(K_2+x))

        if(func_form == 'exp'):
            K_1 = self.demand_mean[i]*3
            K_2 = 1/(self.demand_mean[i])*math.log(1-(self.demand_mean[i]/(K_1*self.processing_time_mean[i])))
            return lambda x: min(x, K_1*(1-math.exp(K_2*x)))
        
        if(func_form == 'poly'):
            C = self.demand_mean[i]*3
            x = self.demand_mean[i]
            y = self.demand_mean[i]/self.processing_time_mean[i]
            k = -x + y + C*x/y - C
            return lambda x: min(x, 0.5*(C+k+x-math.sqrt(C**2+2*C*k+k**2-2*C*x+2*k*x+x**2)))

    def init_prod_state(self, thread_num):
        prod_states = []
        for i in range(self.num_product):
            t = np.array([int(self.demand_mean[i])*self.processing_time_mean[i]])
            prod_states.append(t)
        
        thread_prod_states = []
        for i in range(self.num_product):
            thread_prod_states.append(np.concatenate([np.array([prod_states[i]])]*thread_num))
        
        return thread_prod_states
    
    def init_demand_state(self, thread_num):
        return None
    
    def update_prod_state(self, old_s, Qs, step_num, thread_num):
        
        '''
        for i in range(self.num_product):
            with open('./'+str(i)+'.txt', 'a+') as f:
                for j in range(thread_num):
                    f.write(str(old_s[i][j][0] + Qs[j][i]))
                    f.write('\n')
        '''

        s = []
        production = np.zeros([thread_num, self.num_product])
        for i in range(self.num_product):
            for t in range(thread_num):
                production[t][i] = int(self.clear_functions[i](old_s[i][t][0] + Qs[t][i]))
            new_s = old_s[i][:,0] + Qs[:,i] - production[:,i]
            s.append(new_s.reshape([thread_num, 1]))
        return s, np.stack(production, axis=0)
    
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
        