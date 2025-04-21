import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from envs.SMLP import Env
import numpy as np
from scipy import stats
from scipy import optimize
from copy import deepcopy as dc
from pathlib import Path
import pandas as pd
import os
import time
from matplotlib import pyplot as plt

def evaluate_policies(eval_episodes, episode_max_step, env, Qs, policy_name, instance_id, exp_name):

    costs = []
    holding_costs = []
    backlog_costs = []
    set_up_costs = []

    env.reset(train = False)
    for step in range(episode_max_step):
        state, rewards, sub_agent_done, sub_agent_info = env.step(Qs[:,step].reshape(eval_episodes, env.agent_num), denormalize = False)
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
    
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] +"/results/") / "SMLP" / exp_name / "evaluation" / instance_id
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    f_pth = str(run_dir) + '/' + policy_name + ".csv"
    means = np.array([np.mean(costs), np.mean(holding_costs), np.mean(backlog_costs), np.mean(set_up_costs)])
    stds = np.array([np.std(costs), np.std(holding_costs), np.std(backlog_costs), np.std(set_up_costs)])
    d = np.stack([means, stds], axis=1)
    d = pd.DataFrame(d, columns = ['mean', 'std'], index = ['cost', 'holding cost', 'backlog cost', 'set up cost'])
    d.to_csv(f_pth)

def cf_linearization(cf, x_max, num_points, num_lines):

    y_bound = cf(1e10)
    points = [x_max*i/num_points for i in range(num_points)]

    def obj(x, *params):
        cf, points, y_bound = params
        res = 0
        for p in points:
            t = np.inf
            for i in range(int(x.shape[0]/2)):
                t = np.min([t, x[i*2]*p+x[i*2+1]])
            res += (t-cf(p))**2
        return res
    
    ranges = [(0, 1), (0, 0), (0, 0), (y_bound, y_bound)]
    for i in range(num_lines-1):
        ranges.append((0, 1))
        ranges.append((0, y_bound))

    result = optimize.differential_evolution(obj, tuple(ranges), args=(cf, points, y_bound))

    del_indexes = []
    while(True):
        flag = False
        x_ = dc(result.x)
        for index in del_indexes:
            x_[index*2] = 0
            x_[index*2+1] = y_bound*2
        for i in range(num_lines+1):
            if(i in del_indexes):
                continue
            x_1 = dc(x_)
            x_1[i*2] = 0
            x_1[i*2+1] = y_bound*2
            if(result.fun == obj(x_1, cf, points, y_bound)):
                del_indexes.append(i)
                flag = True
                break
        if(flag == False):
            break

    coeff_list = []
    for i in range(num_lines+1):
        if(i in del_indexes):
            continue
        coeff_list.append([result.x[i*2], result.x[i*2+1]])
    coeff_list = sorted(coeff_list, key=lambda x:x[1])
    if(coeff_list[-1][0] != 0):
        coeff_list.append([0, y_bound])

    cfs = [[], []]
    cfs[0].append(0)
    cfs[1].append(0)
    for j in range(len(coeff_list)-1):
        inter_x = (coeff_list[j+1][1]-coeff_list[j][1])/(coeff_list[j][0]-coeff_list[j+1][0])
        inter_y = (coeff_list[j][0]*coeff_list[j+1][1]-coeff_list[j+1][0]*coeff_list[j][1])/(coeff_list[j][0]-coeff_list[j+1][0])
        cfs[0].append(inter_x)
        cfs[1].append(inter_y)
    cfs[0].append(1e6)
    cfs[1].append(y_bound)
        
    return cfs, result.x

def solve_static_model(net, demand_trace, current_inv, initial_wip, linear_cf, episode_max_step, scenario_num):

    Q_dim = episode_max_step*net.num_product
    Y_dim = episode_max_step*net.num_product
    H_dim = episode_max_step*net.num_product*scenario_num
    B_dim = episode_max_step*net.num_final_product*scenario_num
    P_dim = episode_max_step*net.num_product
    W_dim = episode_max_step*net.num_product
    Q_W_dim = np.sum([(len(linear_cf[i][0])-1)*episode_max_step for i in range(net.num_product)])
    Q_W_i_dim = np.sum([(len(linear_cf[i][0])-1)*episode_max_step for i in range(net.num_product)])

    total_dim = Q_dim + Y_dim + H_dim + B_dim + P_dim + W_dim + Q_W_dim + Q_W_i_dim
    max_const_num = 2500
    
    from gurobipy import GRB, Model
    m = Model('static_static_model')
    m.setParam('OutputFlag', False)
    m.setParam('MIPGap', 0.05)
    m.setParam('TimeLimit', 10800)
    m.setParam('Seed', 1)

    x = m.addMVar((total_dim))
    for i in range(total_dim - Q_W_i_dim, total_dim):
        x[i].VType = GRB.BINARY
    
    for i in range(Q_dim, Q_dim + Y_dim):
        x[i].VType = GRB.BINARY

    c = np.zeros([total_dim])
    for i in range(Q_dim, Q_dim + Y_dim):
        c[i] = net.es[int((i-Q_dim)/episode_max_step)]*scenario_num
    for sce in range(scenario_num):
        for i in range(Q_dim + Y_dim + sce*net.num_product*episode_max_step, Q_dim + Y_dim + (sce+1)*net.num_product*episode_max_step):
            c[i] = net.hs[int((i-Q_dim-Y_dim-sce*net.num_product*episode_max_step)/episode_max_step)]
    for sce in range(scenario_num):
        for i in range(Q_dim + Y_dim + H_dim + sce*net.num_final_product*episode_max_step, Q_dim + Y_dim + H_dim + (sce+1)*net.num_final_product*episode_max_step):
            c[i] = net.bs[int((i-Q_dim-Y_dim-H_dim-sce*net.num_final_product*episode_max_step)/episode_max_step)]
    
    m.setObjective(c @ x, GRB.MINIMIZE)

    A_1 = [] # =
    b_1 = []
    for sce in range(scenario_num):
        for i in range(net.num_product):
            t_ = total_dim - W_dim - Q_W_dim - Q_W_i_dim + i*episode_max_step
            arr = np.zeros([total_dim])
            arr[t_] = 1
            A_1.append(arr)
            b_1.append(initial_wip[i])
            if(len(A_1) == max_const_num):
                m.addConstr(np.array(A_1) @ x == np.array(b_1))
                del A_1, b_1
                A_1 = []
                b_1 = []
                
    if(len(A_1) > 0):
        m.addConstr(np.array(A_1) @ x == np.array(b_1))
    del A_1, b_1
    
    A_2 = [] # = 
    b_2 = []
    for sce in range(scenario_num):
        for i in range(net.num_final_product):
            for t in range(episode_max_step):
                arr = np.zeros([total_dim])
                if(t != 0):
                    s = total_dim - P_dim - W_dim - Q_W_dim - Q_W_i_dim + i*episode_max_step
                    arr[s:s+t] = 1
                s = Q_dim + Y_dim + i*episode_max_step + sce*net.num_product*episode_max_step
                arr[s+t] = -1
                s = Q_dim + Y_dim + H_dim + i*episode_max_step + sce*net.num_final_product*episode_max_step
                arr[s+t] = 1
                b = np.sum(demand_trace[sce,:t+1,i]) - current_inv[i]
                A_2.append(arr)
                b_2.append(b)
                if(len(A_2) == max_const_num):
                    m.addConstr(np.array(A_2) @ x == np.array(b_2))
                    del A_2, b_2
                    A_2 = []
                    b_2 = []
    
    if(len(A_2) > 0):
        m.addConstr(np.array(A_2) @ x == np.array(b_2))
    del A_2, b_2
    
    A_3 = [] # = 
    b_3 = []
    for sce in range(scenario_num):
        for i in range(net.num_final_product, net.num_product):
            for t in range(episode_max_step):
                arr = np.zeros([total_dim])
                if(t != 0):
                    s = total_dim - P_dim - W_dim - Q_W_dim - Q_W_i_dim + i*episode_max_step
                    arr[s:s+t] = 1
                for ii in range(net.num_product):
                    s = ii*episode_max_step
                    arr[s:s+t+1] = arr[s:s+t+1] - net.M[i][ii]
                s = Q_dim + Y_dim + i*episode_max_step + sce*net.num_product*episode_max_step
                arr[s+t] = -1
                b = - current_inv[i]
                A_3.append(arr)
                b_3.append(b)
                if(len(A_3) == max_const_num):
                    m.addConstr(np.array(A_3) @ x == np.array(b_3))
                    del A_3, b_3
                    A_3 = []
                    b_3 = []
    
    if(len(A_3) > 0):
        m.addConstr(np.array(A_3) @ x == np.array(b_3))
        
    del A_3, b_3

    A_4 = [] # <= 
    b_4 = []
    for i in range(net.num_product):
        for t in range(episode_max_step):
            arr = np.zeros([total_dim])
            s = i*episode_max_step
            arr[s+t] = 1
            s = Q_dim + i*episode_max_step
            arr[s+t] = -1e9
            A_4.append(arr)
            b_4.append(0)
            if(len(A_4) == max_const_num):
                m.addConstr(np.array(A_4) @ x <= np.array(b_4))
                del A_4, b_4
                A_4 = []
                b_4 = []
    if(len(A_4) > 0):
        m.addConstr(np.array(A_4) @ x <= np.array(b_4))
    del A_4, b_4

    A_5 = [] # <=
    b_5 = []
    for i in range(net.num_product):
        for t in range(0, episode_max_step):
            arr = np.zeros([total_dim])
            s = Q_dim + i*episode_max_step
            arr[s] = 1
            A_5.append(arr)
            b_5.append(1)
            if(len(A_5) == max_const_num):
                m.addConstr(np.array(A_5) @ x <= np.array(b_5))
                del A_5, b_5
                A_5 = []
                b_5 = []
    
    if(len(A_5) > 0):
        m.addConstr(np.array(A_5) @ x <= np.array(b_5))
    del A_5, b_5

    A_6 = [] # ==
    b_6 = []
    for i in range(net.num_product):
        for t in range(1, episode_max_step):
            arr = np.zeros([total_dim])
            s = total_dim - W_dim - Q_W_dim  - Q_W_i_dim + i*episode_max_step
            arr[s+t] = 1
            arr[s+t-1] = -1
            s = i*episode_max_step
            arr[s+t-1] = -1
            s = total_dim - P_dim - W_dim - Q_W_dim  - Q_W_i_dim + i*episode_max_step
            arr[s+t-1] = 1
            b = 0
            A_6.append(arr)
            b_6.append(b)
            if(len(A_6) == max_const_num):
                m.addConstr(np.array(A_6) @ x == np.array(b_6))
                del A_6, b_6
                A_6 = []
                b_6 = []
    if(len(A_6) > 0):
        m.addConstr(np.array(A_6) @ x == np.array(b_6))
    del A_6, b_6

    A_7 = [] # ==
    b_7 = []
    for i in range(net.num_product):
        for t in range(episode_max_step):
            arr = np.zeros([total_dim])
            s = total_dim - W_dim - Q_W_dim  - Q_W_i_dim + i*episode_max_step
            arr[s+t] = -1
            s = i*episode_max_step
            arr[s+t] = -1
            if(i == 0):
                s = total_dim - Q_W_dim  - Q_W_i_dim
            else:
                s = total_dim - Q_W_dim  - Q_W_i_dim + np.sum([(len(linear_cf[ii][0])-1)*episode_max_step for ii in range(i)])
            for ii in range(len(linear_cf[i][0])-1):
                ss = s + ii*episode_max_step
                arr[ss+t] = 1
            b = 0
            A_7.append(arr)
            b_7.append(b)
            if(len(A_7) == max_const_num):
                m.addConstr(np.array(A_7) @ x == np.array(b_7))
                del A_7, b_7
                A_7 = []
                b_7 = []
    if(len(A_7) > 0):
        m.addConstr(np.array(A_7) @ x == np.array(b_7))
    del A_7, b_7

    A_8 = [] # >=
    b_8 = []
    for i in range(net.num_product):
        if(i==0):
            s = total_dim - Q_W_dim  - Q_W_i_dim
            ss = total_dim - Q_W_i_dim
        else:
            s = total_dim - Q_W_dim  - Q_W_i_dim + np.sum([(len(linear_cf[ii][0])-1)*episode_max_step for ii in range(i)])
            ss = total_dim - Q_W_i_dim + np.sum([(len(linear_cf[ii][0])-1)*episode_max_step for ii in range(i)])
        for t in range(episode_max_step):
            for l in range(len(linear_cf[i][0])-1):
                arr = np.zeros([total_dim])
                s_ = s + l*episode_max_step
                arr[s_+t] = 1
                s_ = ss + l*episode_max_step
                arr[s_+t] = -linear_cf[i][0][l]
                b = 0
                A_8.append(arr)
                b_8.append(b)
                if(len(A_8) == max_const_num):
                    m.addConstr(np.array(A_8) @ x >= np.array(b_8))
                    del A_8, b_8
                    A_8 = []
                    b_8 = []
    if(len(A_8) > 0):
        m.addConstr(np.array(A_8) @ x >= np.array(b_8))
    del A_8, b_8

    A_9 = [] # <=
    b_9 = []
    for i in range(net.num_product):
        if(i==0):
            s = total_dim - Q_W_dim  - Q_W_i_dim
            ss = total_dim - Q_W_i_dim
        else:
            s = total_dim - Q_W_dim  - Q_W_i_dim + np.sum([(len(linear_cf[ii][0])-1)*episode_max_step for ii in range(i)])
            ss = total_dim - Q_W_i_dim + np.sum([(len(linear_cf[ii][0])-1)*episode_max_step for ii in range(i)])
        for t in range(episode_max_step):
            for l in range(len(linear_cf[i][0])-1):
                arr = np.zeros([total_dim])
                s_ = s + l*episode_max_step
                arr[s_+t] = 1
                s_ = ss + l*episode_max_step
                arr[s_+t] = -linear_cf[i][0][l+1]
                b = 0
                A_9.append(arr)
                b_9.append(b)
                if(len(A_9) == max_const_num):
                    m.addConstr(np.array(A_9) @ x <= np.array(b_9))
                    del A_9, b_9
                    A_9 = []
                    b_9 = []
    if(len(A_9) > 0):
        m.addConstr(np.array(A_9) @ x <= np.array(b_9))
    del A_9, b_9

    A_10 = [] # ==
    b_10 = []
    for i in range(net.num_product):
        if(i==0):
            s = total_dim - Q_W_dim  - Q_W_i_dim
            ss = total_dim - Q_W_i_dim
        else:
            s = total_dim - Q_W_dim  - Q_W_i_dim + np.sum([(len(linear_cf[ii][0])-1)*episode_max_step for ii in range(i)])
            ss = total_dim - Q_W_i_dim + np.sum([(len(linear_cf[ii][0])-1)*episode_max_step for ii in range(i)])
        for t in range(episode_max_step):
            arr = np.zeros([total_dim])
            s_ = total_dim - P_dim - W_dim - Q_W_dim - Q_W_i_dim + i*episode_max_step + t
            arr[s_] = -1
            for l in range(len(linear_cf[i][0])-1):
                ratio = (linear_cf[i][1][l+1] - linear_cf[i][1][l])/(linear_cf[i][0][l+1] - linear_cf[i][0][l])
                s_ = s + l*episode_max_step
                arr[s_+t] = ratio
                s_ = ss + l*episode_max_step
                arr[s_+t] = linear_cf[i][1][l] - ratio*linear_cf[i][0][l]
            b = 0
            A_10.append(arr)
            b_10.append(b)
            if(len(A_10) == max_const_num):
                m.addConstr(np.array(A_10) @ x == np.array(b_10))
                del A_10, b_10
                A_10 = []
                b_10 = []
    if(len(A_10) > 0):
        m.addConstr(np.array(A_10) @ x == np.array(b_10))
    del A_10, b_10

    A_11 = [] # ==
    b_11 = []
    for i in range(net.num_product):
        if(i==0):
            ss = total_dim - Q_W_i_dim
        else:
            ss = total_dim - Q_W_i_dim + np.sum([(len(linear_cf[ii][0])-1)*episode_max_step for ii in range(i)])
        for t in range(episode_max_step):
            arr = np.zeros([total_dim])
            for l in range(len(linear_cf[i][0])-1):
                s_ = ss + l*episode_max_step
                arr[s_+t] = 1
            b = 1
            A_11.append(arr)
            b_11.append(b)
            if(len(A_11) == max_const_num):
                m.addConstr(np.array(A_11) @ x == np.array(b_11))
                del A_11, b_11
                A_11 = []
                b_11 = []
    if(len(A_11) > 0):
        m.addConstr(np.array(A_11) @ x == np.array(b_11))
    del A_11, b_11

    m.optimize()
    opt_vars = []
    for v in m.getVars():
        opt_vars.append(v.x)

    Qs = [[] for i in range(episode_max_step)]
    for t in range(episode_max_step):
        for i in range(net.num_product):
            s = i*episode_max_step
            Qs[t].append(int(opt_vars[s+t]))
    Qs = np.array([np.array(a) for a in Qs])
    del m
    return Qs

def OPT(seed, data_pth, instance_id, episode_max_step, eval_episodes, exp_name = 'Clearing_Function_Frac'):
    
    env = Env(seed, data_pth, instance_id, 0, episode_max_step, thread_num = eval_episodes, exp_name = exp_name)
    net = env.prod_net
    tr = []
    env.reset(train = False)

    for step in range(episode_max_step):
        _, _, _, infos = env.step(np.zeros([eval_episodes, env.agent_num]), denormalize = False)
        tr.append(infos[0]['demands'][:,:net.num_final_product])
    tr = np.stack(tr, axis=1)
    
    current_inv = dc(net.initial_inventory)
    initial_wip = [o[0,0] for o in net.init_prod_state(thread_num=1)]
    l_cfs = []
    for i in range(env.agent_num):
        num_lines = 3
        num_points = 100
        x_max = net.demand_mean[i]*10
        cfs, coeffs = cf_linearization(net.clear_functions[i], x_max, num_points, num_lines)
        l_cfs.append(cfs)
    
    Qss = []
    for i in range(eval_episodes):
        print(i)
        Qs = solve_static_model(net, tr[i,:,:].reshape(1, tr.shape[1], tr.shape[2]), current_inv, initial_wip, l_cfs, episode_max_step, 1)
        Qss.append(Qs)
    Qss = np.stack(Qss, axis=0)
    
    env = Env(seed, data_pth, instance_id, 0, episode_max_step, thread_num = eval_episodes, exp_name = exp_name)
    evaluate_policies(eval_episodes, episode_max_step, env, Qss, 'OPT', instance_id, exp_name = exp_name)

OPT(1, './data/MSOM-06-038-R2.xls', '01', 50, 100, exp_name = 'Clearing_Function_Exp')
OPT(1, './data/MSOM-06-038-R2.xls', '01', 50, 100, exp_name = 'Clearing_Function_Poly')
OPT(1, './data/MSOM-06-038-R2.xls', '01', 50, 100, exp_name = 'Clearing_Function_Frac')