import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from envs.SMLP_SCHED import Env
import numpy as np
from scipy import stats
from copy import deepcopy as dc
from pathlib import Path
import pandas as pd
import os
import time

def estimate_lead_time(net):
    lead_times = [3 for i in range(net.num_product)]
    return lead_times

def compute_service_level_based_max_demand(net, i, t_interval):
    if(t_interval == 0):
        return 0
    if(i < net.num_final_product):
        sl = net.bs[i]/(net.hs[i] + net.bs[i])
        mean = net.demand_mean[i]*t_interval
        std = net.demand_std[i]*(t_interval**0.5)
        s = int(stats.gamma.ppf(sl, a = mean**2/(std**2), loc = 0, scale = (std**2)/mean))
        return s
    else:
        s = 0
        for j in range(net.num_product):
            if(net.M[i][j] > 0):
                s += net.M[i][j]*compute_service_level_based_max_demand(net, j, t_interval)
        return s

def compute_largest_lead_time(net, i, lead_times):

    if(np.sum(net.M[:, i]) == 0):
        return 0
    t = []
    for j in range(net.num_product):
        if(net.M[j][i] > 0):
            t.append(compute_largest_lead_time(net, j, lead_times) + lead_times[j] + 1)
    return np.max(t)

def compute_period_demand(net, i, t, orders):
    
    if(i < net.num_final_product):
        return net.demand_mean[i]
    else:
        demand = 0
        for j in range(net.num_product):
            if(net.M[i][j] > 0):
                demand += net.M[i][j]*orders[j][t]
        return demand

def evaluate_policies(eval_episodes, episode_max_step, env, Qs, policy_name, instance_id, policy):

    costs = []
    holding_costs = []
    backlog_costs = []
    set_up_costs = []

    env.reset(train = False)
    for step in range(episode_max_step):
        planning_a = Qs[:,step].reshape(eval_episodes, env.agent_num, 1)
        scheduling_a = np.zeros([eval_episodes, env.agent_num, 5])
        scheduling_a[:,:,policy[1]] = 1
        a = np.concatenate([planning_a, scheduling_a], axis = 2)
        state, rewards, sub_agent_done, sub_agent_info = env.step(a, denormalize = False)
        costs.append(np.sum(rewards, axis=1))
        holding_costs.append(np.sum(sub_agent_info[0]['holding costs'], axis=1))
        backlog_costs.append(np.sum(sub_agent_info[0]['backlog costs'], axis=1))
        set_up_costs.append(np.sum(sub_agent_info[0]['set up costs'], axis=1))
    costs = np.mean(np.stack(costs, axis=0), axis=0)
    holding_costs = np.mean(np.stack(holding_costs, axis=0), axis=0)
    backlog_costs = np.mean(np.stack(backlog_costs, axis=0), axis=0)
    set_up_costs = np.mean(np.stack(set_up_costs, axis=0), axis=0)
    print('Results for ' + policy_name + ' :')
    print('overall costs ', 'mean: ', np.mean(costs), ' std: ', np.std(costs))
    print('holding costs ', 'mean: ', np.mean(holding_costs), ' std: ', np.std(holding_costs))
    print('backlog costs ', 'mean: ', np.mean(backlog_costs), ' std: ', np.std(backlog_costs))
    print('set up costs ', 'mean: ', np.mean(set_up_costs), ' std: ', np.std(set_up_costs))
    
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] +"/results/") / "SMLP_SCHED" / "Flow_Shop" / "evaluation" / instance_id
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    f_pth = str(run_dir) + '/' + policy_name + '_' + policy[0] + ".csv"
    means = np.array([np.mean(costs), np.mean(holding_costs), np.mean(backlog_costs), np.mean(set_up_costs)])
    stds = np.array([np.std(costs), np.std(holding_costs), np.std(backlog_costs), np.std(set_up_costs)])
    d = np.stack([means, stds], axis=1)
    d = pd.DataFrame(d, columns = ['mean', 'std'], index = ['cost', 'holding cost', 'backlog cost', 'set up cost'])
    d.to_csv(f_pth)

def GW(seed, data_pth, instance_id, episode_max_step):
    
    H = 10
    si_set = [i*2 for i in range(H)]
    s_set = [i*2 for i in range(H)]

    env = Env(seed, data_pth, instance_id, 0, episode_max_step, exp_name = 'Flow_Shop')
    net = env.prod_net
    lead_times = estimate_lead_time(net)
    c = []

    for i in range(env.agent_num):
        for si in range(H):
            for s in range(H):
                if(lead_times[i]+si_set[si]<s_set[s]):
                    c.append(0)
                else:
                    t = net.hs[i]*(compute_service_level_based_max_demand(net, i, lead_times[i]+si_set[si]-s_set[s]) - net.demand_mean[i]*(lead_times[i]+si_set[si]-s_set[s]))
                    c.append(t)
    c = np.array(c)

    A_1 = np.zeros([env.agent_num, env.agent_num*H*H])
    for i in range(env.agent_num):
        for j in range(H*H):
            A_1[i][j+i*H*H] = 1
    b_1 = np.array([1 for i in range(env.agent_num)])

    A_2 = []
    for i in range(env.agent_num):
        for si in range(H):
            for s in range(H):
                if(lead_times[i]+si_set[si]<s_set[s]):
                    t = np.zeros([env.agent_num*H*H])
                    t[i*H*H + H*si + s] = 1
                    A_2.append(t)
    A_2 = np.array(A_2)
    b_2 = np.array([0 for i in range(A_2.shape[0])])

    A_3 = []
    for i in range(env.agent_num):
        for j in range(env.agent_num):
            if(net.M[i][j] > 0):
                for si_j in range(H):
                    for s_i in range(H):
                        if(s_i <= si_j):
                            continue
                        for s_j in range(H):
                            for si_i in range(H):
                                t = np.zeros([env.agent_num*H*H])
                                t[i*H*H+si_i*H+s_i] = 1
                                t[j*H*H+si_j*H+s_j] = 1
                                A_3.append(t)
    A_3 = np.array(A_3)
    b_3 = np.array([1 for i in range(A_3.shape[0])])

    A_4 = []
    for i in range(env.final_product_num):
        for si in range(H):
            for s in range(H):
                if(s == 0):
                    continue
                t = np.zeros([env.agent_num*H*H])
                t[i*H*H + si*H + s] = 1
                A_4.append(t)

    A_4 = np.array(A_4)
    b_4 = np.array([0 for i in range(A_4.shape[0])])

    A = np.concatenate([A_1, A_2, A_3, A_4], axis = 0)
    b = np.concatenate([b_1, b_2, b_3, b_4], axis = 0)

    import cvxpy as cp

    x = cp.Variable(env.agent_num*H*H, integer = True)
    obj = cp.Minimize(c@x)
    cons = [A_1@x == b_1, A_2@x == b_2, A_3@x <= b_3, A_4@x == b_4, x>=0, x<=1]
    prob = cp.Problem(obj, cons)
    prob.solve(solver = 'SCIPY', verbose = False)
    solution = []
    for i in range(x.value.shape[0]):
        if(x.value[i] > 0.5):
            solution.append(i)
    safety_stocks = []

    for i in range(len(solution)):
        t = solution[i] - i*H*H
        si = int(t/H)
        s = t%H
        ss = compute_service_level_based_max_demand(net, i, lead_times[i]+si_set[si]-s_set[s]) - net.demand_mean[i]*(lead_times[i]+si_set[si]-s_set[s])
        safety_stocks.append(ss)
    
    return safety_stocks

def EOQ(seed, data_pth, instance_id, episode_max_step, eval_episodes, safety_stocks, policy):

    env = Env(seed, data_pth, instance_id, 0, episode_max_step, thread_num = eval_episodes, exp_name = 'Flow_Shop')
    net = env.prod_net
    lead_times = estimate_lead_time(net)

    EOQs = []
    largest_lead_times = [compute_largest_lead_time(net, i, lead_times) for i in range(net.num_product)]

    for i in range(env.agent_num):
        EOQs.append(int((2*net.es[i]*net.demand_mean[i]/net.hs[i])**0.5))

    orders = [[] for i in range(env.agent_num)]
    
    for i in range(env.agent_num):
        for step in range(episode_max_step):
            demand = 0
            if(i < env.final_product_num):
                demand = net.demand_mean[i]*(step+1)
            else:
                for j in range(env.agent_num):
                    if(env.M[i][j] > 0):
                        demand += env.M[i][j]*np.sum(orders[j][:step+1])
            IL = np.sum(orders[i]) - demand
            
            m = -1
            while(True):
                m += 1
                if(m*EOQs[i] + IL > safety_stocks[i]):
                    break
            orders[i].append(int(m*EOQs[i]))
    
    Qs = [[] for i in range(episode_max_step)]
    unmet = [0 for i in range(env.agent_num)]
    
    for i in range(env.agent_num):
        for step in range(episode_max_step):
            if(step <= largest_lead_times[i] + 1 and np.sum(net.M[:, i]) > 0):
                unmet[i] += orders[i][step]
                Qs[step].append(unmet[i])
                unmet[i] -= net.demand_mean[i]
            else:
                Qs[step].append(orders[i][step])

    Qs = np.array([np.array(a) for a in Qs])
    Qs = [Qs for _ in range(eval_episodes)]
    Qs = np.stack(Qs, axis=0)

    evaluate_policies(eval_episodes, episode_max_step, env, Qs, 'EOQ', instance_id, policy)

def EOP(seed, data_pth, instance_id, episode_max_step, eval_episodes, safety_stocks, policy):

    env = Env(seed, data_pth, instance_id, 0, episode_max_step, thread_num = eval_episodes, exp_name = 'Flow_Shop')
    net = env.prod_net
    lead_times = estimate_lead_time(net)

    EOPs = []
    largest_lead_times = [compute_largest_lead_time(net, i, lead_times) for i in range(net.num_product)]

    for i in range(env.agent_num):
        EOPs.append(int((2*net.es[i]/(net.hs[i]*net.demand_mean[i]))**0.5))
    
    orders = [[] for i in range(env.agent_num)]
    for i in range(env.agent_num):
        last_production = -1
        for step in range(episode_max_step):
            demand = 0
            if(i < env.final_product_num):
                demand = net.demand_mean[i]*(step+1)
            else:
                for j in range(env.agent_num):
                    if(env.M[i][j] > 0):
                        demand += env.M[i][j]*np.sum(orders[j][:step+1])
            IL = np.sum(orders[i]) - demand

            t = 0
            if((last_production==-1 and IL <= safety_stocks[i]) or (last_production >= 0 and last_production + EOPs[i] == step)):
                last_production = step
                if(i < env.final_product_num):
                    t = net.demand_mean[i]*EOPs[i]
                else:
                    demand = 0
                    for j in range(env.agent_num):
                        if(env.M[i][j] > 0):
                            demand += env.M[i][j]*np.sum(orders[j][step+1:step+EOPs[i]+1])
                    t = demand
            t = t + safety_stocks[i] - IL if step == 0 else t
            orders[i].append(max(int(t), 0))

    Qs = [[] for i in range(episode_max_step)]
    unmet = [0 for i in range(env.agent_num)]
    
    for i in range(env.agent_num):
        for step in range(episode_max_step):
            if(step <= largest_lead_times[i] + 1 and np.sum(net.M[:, i]) > 0):
                unmet[i] += orders[i][step]
                Qs[step].append(unmet[i])
                unmet[i] -= net.demand_mean[i]
            else:
                Qs[step].append(orders[i][step])

    Qs = np.array([np.array(a) for a in Qs])
    Qs = [Qs for _ in range(eval_episodes)]
    Qs = np.stack(Qs, axis=0)

    evaluate_policies(eval_episodes, episode_max_step, env, Qs, 'EOP', instance_id, policy)

def SM(seed, data_pth, instance_id, episode_max_step, eval_episodes, safety_stocks, policy):

    env = Env(seed, data_pth, instance_id, 0, episode_max_step, thread_num = eval_episodes, exp_name = 'Flow_Shop')
    net = env.prod_net
    lead_times = estimate_lead_time(net)
    largest_lead_times = [compute_largest_lead_time(net, i, lead_times) for i in range(net.num_product)]
        
    orders = [[] for i in range(env.agent_num)]
    for i in range(env.agent_num):
        for step in range(episode_max_step):
            demand = 0
            if(i < env.final_product_num):
                demand = net.demand_mean[i]*(step+1)
            else:
                for j in range(env.agent_num):
                    if(env.M[i][j] > 0):
                        demand += env.M[i][j]*np.sum(orders[j][:step+1])
            IL = np.sum(orders[i]) - demand
            
            if(IL <= safety_stocks[i]):
                P = 0
                last = np.inf
                while(True):
                    if(P + step + 1 >= episode_max_step):
                        break
                    P += 1
                    s = np.sum([(t+1)*net.hs[i]*compute_period_demand(net, i, step+t+1, orders) for t in range(P)])
                    f = (net.es[i] + s)/P
                    if(f > last):
                        P -= 1
                        break
                    last = f

                if(step == 0):
                    tt = np.sum([compute_period_demand(net, i, step+t+1, orders) for t in range(P)]) + safety_stocks[i] - IL
                else:
                    tt = np.sum([compute_period_demand(net, i, step+t+1, orders) for t in range(P)])
                
                orders[i].append(int(tt))
            else:
                orders[i].append(0)
    
    Qs = [[] for i in range(episode_max_step)]
    unmet = [0 for i in range(env.agent_num)]
    
    for i in range(env.agent_num):
        for step in range(episode_max_step):
            if(step <= largest_lead_times[i] + 1 and np.sum(net.M[:, i]) > 0):
                unmet[i] += orders[i][step]
                Qs[step].append(unmet[i])
                unmet[i] -= net.demand_mean[i]
            else:
                Qs[step].append(orders[i][step])

    Qs = np.array([np.array(a) for a in Qs])
    Qs = [Qs for _ in range(eval_episodes)]
    Qs = np.stack(Qs, axis=0)

    evaluate_policies(eval_episodes, episode_max_step, env, Qs, 'SM', instance_id, policy)

def L4L(seed, data_pth, instance_id, episode_max_step, eval_episodes, safety_stocks, policy):

    env = Env(seed, data_pth, instance_id, 0, episode_max_step, thread_num = eval_episodes, exp_name = 'Flow_Shop')
    net = env.prod_net
    lead_times = estimate_lead_time(net)
    largest_lead_times = [compute_largest_lead_time(net, i, lead_times) for i in range(net.num_product)]
    
    orders = [[] for i in range(env.agent_num)]
    for i in range(env.agent_num):
        for step in range(episode_max_step):
            demand = 0
            if(i < env.final_product_num):
                demand = net.demand_mean[i]*(step+1)
            else:
                for j in range(env.agent_num):
                    if(env.M[i][j] > 0):
                        demand += env.M[i][j]*np.sum(orders[j][:step+1])
            IL = np.sum(orders[i]) - demand
            t = net.demand_mean[i] + safety_stocks[i] - IL
            orders[i].append(max(int(t), 0))
    
    Qs = [[] for i in range(episode_max_step)]
    unmet = [0 for i in range(env.agent_num)]
    
    for i in range(env.agent_num):
        for step in range(episode_max_step):
            if(step <= largest_lead_times[i] + 1 and np.sum(net.M[:, i]) > 0):
                unmet[i] += orders[i][step]
                Qs[step].append(unmet[i])
                unmet[i] -= net.demand_mean[i]
            else:
                Qs[step].append(orders[i][step])
    
    Qs = np.array([np.array(a) for a in Qs])
    Qs = [Qs for _ in range(eval_episodes)]
    Qs = np.stack(Qs, axis=0)

    evaluate_policies(eval_episodes, episode_max_step, env, Qs, 'L4L', instance_id, policy)

def SS(seed, data_pth, instance_id, episode_max_step, eval_episodes, policy):

    env = Env(seed, data_pth, instance_id, 0, episode_max_step, thread_num = eval_episodes, exp_name = 'Flow_Shop')
    net = env.prod_net
    lead_times = estimate_lead_time(net)

    def compute_ss(net, product, service_level, lead_times, scaled_mean=100, max_value_to_mean_ratio=10):
    
        def G_func(x, p, service_level):
            H = 1
            B = 1/(1/service_level-1)
            e = 0
            for i in range(500):
                e += p(i)*(np.max([x-i,0])*H + np.max([i-x,0])*B)
            return e

        scale = net.demand_mean[product]*(lead_times[product]+1)/scaled_mean
        mean = net.demand_mean[product]*(lead_times[product]+1)/scale
        std = net.demand_std[product]*((lead_times[product]+1)**0.5)/scale
        max_value = int(mean*max_value_to_mean_ratio)

        p = lambda x: stats.gamma.cdf(x+1, a = mean**2/(std**2), loc = 0, scale = (std**2)/mean) - stats.gamma.cdf(x, a = mean**2/(std**2), loc = 0, scale = (std**2)/mean)
        G = {}
        for n in range(-max_value, max_value):
            G[n] = G_func(n, p, service_level)

        ms = [1/(1-p(0))]
        Ms = [0]
        for j in range(1, max_value):
            res = 0
            Ms.append(Ms[-1] + ms[-1])
            for i in range(1, j+1):
                res += p(i)*ms[j-i]
            ms.append(res/(1-p(0)))

        c = lambda s, S: (1/Ms[S-s])*(net.es[product]/scale+np.sum([ms[j]*G[S-j] for j in range(S-s)]))

        y_s = np.argmin([G[j] for j in range(max_value)])
        s_0 = y_s
        S_0 = y_s
        j = 0
        while(True):
            j += 1
            if(c(s_0-j, S_0) <= G[s_0-j]):
                s = s_0 - j
                s_0 = dc(s)
                c_0 = c(s, S_0)
                S = S_0 + 1
                break
        
        while(True):
            if(G[S] > c_0):
                break
            if(c(s, S) < c_0):
                S_0 = dc(S)
                while(c(s, S_0) <= G[s+1]):
                    s += 1
                c_0 = c(s, S_0)
            S += 1
        return int(s*scale), int(S*scale)
    
    '''
    s_set = []
    S_set = []
    for i in range(env.agent_num):
        if(i < net.num_final_product):
            service_level = net.bs[i]/(net.bs[i]+net.hs[i])
        else:
            service_level = 0.99
        s, S = compute_ss(net, i, service_level, lead_times)
        s_set.append(s)
        S_set.append(S)
    '''
    s_set = [99, 17, 28, 125, 47, 170, 170, 170]                                                                                                   
    S_set = [167, 27, 45, 199, 73, 270, 270, 270]
    print(s_set)
    print(S_set)

    Qs = np.zeros([eval_episodes, episode_max_step, env.agent_num, 1])
    env.reset(train = False)
    for step in range(episode_max_step):
        for k in range(eval_episodes):
            for i in range(env.agent_num):
                if(i < net.num_final_product):
                    IL = env.inventory[k,i] - env.backlog[k,i] + np.sum(env.prod_states[i][k][::5])
                else:
                    IL = env.inventory[k,i] - env.lost_sales[k,i-net.num_final_product] + np.sum(env.prod_states[i][k][::5])
                if(IL <= s_set[i]):
                    Qs[k][step][i] = S_set[i]-IL
        
        scheduling_a = np.zeros([eval_episodes, episode_max_step, env.agent_num, 5])
        scheduling_a[:,:,:,policy[1]] = 1
        a = np.concatenate([Qs, scheduling_a], axis = 3)
                
        state, rewards, sub_agent_done, sub_agent_info = env.step(a[:,step,:].reshape(eval_episodes, env.agent_num, 6), denormalize = False)
    
    env = Env(seed, data_pth, instance_id, 0, episode_max_step, thread_num = eval_episodes, exp_name = 'Flow_Shop')
    evaluate_policies(eval_episodes, episode_max_step, env, Qs, 'ss', instance_id, policy)

scheduling_policies = [['SPT', 0], ['LPT', 1], ['LWKR', 2], ['MWKR', 3], ['AT-RPT', 4]]

for policy in scheduling_policies:
    ss = GW(1, './data/MSOM-06-038-R2.xls', '01', 100)
    EOP(1, './data/MSOM-06-038-R2.xls', '01', 100, 100, ss, policy)
    SM(1, './data/MSOM-06-038-R2.xls', '01', 100, 100, ss, policy)
    EOQ(1, './data/MSOM-06-038-R2.xls', '01', 100, 100, ss, policy)
    L4L(1, './data/MSOM-06-038-R2.xls', '01', 100, 100, ss, policy)
    SS(1, './data/MSOM-06-038-R2.xls', '01', 100, 100, policy)