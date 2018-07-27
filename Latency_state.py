# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 23:04:31 2018

@author: xuan
"""
import sys
from gurobipy import Model, GRB
import matplotlib.pyplot as plt
import pandas as pd
import random
import math

# product function
def prod(x):
    product = 1
    for i in x:
        product = product * i
    return product
# access node generator
def access_node_generator(max_x, max_y, an):
    an_topo = {}
    t = 0
    for i in range(max_x):
        for j in range(max_y):
            an_topo[an[t]] = ((i+0.5)**3, (j+0.5)**3)
            t = t + 1
    return an_topo
# cloud node generator
def cloud_node_generator(max_x, max_y, dc):
    dc_topo = {}
    for t in dc:
        coor = (random.randint(1, max_x), random.randint(1, max_y))
        while(coor in dc_topo.values()):
            coor = (random.randint(1, max_x)**3, random.randint(1, max_y)**3)
        dc_topo[t] = coor
    return dc_topo
    

# define dataframes to store results
df_state = pd.DataFrame(index = range(20), columns = ['Bandwidth','Handover','OST',
                  'OPL','PO'])
df_latency = pd.DataFrame(index = range(20), columns = ['Bandwidth','Handover','OST',
                  'OPL', 'PO'])
df_cost = pd.DataFrame(index = range(20), columns = ['Bandwidth','Handover','OST',
                  'OPL', 'PO'])
df_num_func = pd.DataFrame(index = range(20), columns = ['Bandwidth', 'Handover','OST',
                  'OPL', 'PO'])


# number of cloud centers, access nodes
M_dc = 10
M_an = 30
dc = ['dc' + str(i) for i in range(M_dc)]
an = ['an' + str(i) for i in range(M_an)]

# initialize distance matrix
an_topo = access_node_generator(5, 6, an)
dc_topo = cloud_node_generator(5, 6, dc)
Sd = 200
Lw = 1
Wg = 1
C = {}
for i in an:
    for j in dc:
        C[i,j] = math.hypot(an_topo[i][1] - dc_topo[j][1],an_topo[i][0] - dc_topo[j][0])  

# maximum values for state frequency and latency
Latency_max = 50000
State_max = 50000
################# define optimization models ###########################################
################# optimize state transfer ############################################## 

def optimize_state_transfer(Bw, h):
    # create a model
    m = Model('state transfer')
    # create decision variables
    x = m.addVars(an, an, vtype = GRB.BINARY, name = 'x')
    y = m.addVars(an, dc, vtype = GRB.BINARY, name = 'y')
    
    ######### add constraints ##########
    # x is symmetric
    m.addConstrs((x[i,j] == x[j,i] for i in an for j in an), 'symmetric constraint')
    # Two service areas cant use the same stateful functions when x[i,j] = 0
    m.addConstrs((y[i,t] + y[j,t] <= x[i,j] + 1 for i in an for j in an for t in dc), 'x = 0 constraint')
    # Two service areas use the same stateful functions when x[i,j] = 1
    m.addConstrs(((y[i,t] - y[j,t] <= 1 - x[i,j]) for i in an for j in an for t in dc), 'x = 1 constraint')
    # Each should be at least managed by one dc
    m.addConstrs((sum(y[i,t] for t in dc) == 1 for i in an),'one dc for one access node')
    # high availability constraint
    m.addConstr(sum((1-x[i,j]) for i in an for j in an if i != j) >= 1,'ha')
    # Maximum latency budget
    m.addConstr((sum(C[i,t]*(Sd/Bw + Lw + Wg)*y[i,t] for i in an for t in dc) <= Latency_max), 'max latency constraint')

    ######## Objective function #########
    m.setObjective(sum(h*(1-x[i,j]) for i in an for j in an if i != j), GRB.MINIMIZE)
    
    ######## run model ##################
    m.optimize()

    ######## Calculate performance metrics #########
    # latency 
    latency = sum(getattr(y[i,t],'X')*C[i,t]*(Sd/Bw + Lw + Wg) for i in an for t in dc)
    
    # state transfer frequency
    state_transfer = sum(h*(1-getattr(x[i,j], 'X')) for i in an for j in an if i != j)
    
    # total cost
    cost = latency + state_transfer
    
    # number of function
    num_func = M_dc - sum(prod(1 - getattr(y[i,t], 'X') for i in an) for t in dc) 
    
    # total metrics
    total_metrics = [Bw, h, state_transfer, latency, num_func, cost]
    
    # utopia point, nadir point
    best_state =  state_transfer
    worst_latency =  latency
    
    return total_metrics, best_state, worst_latency

################# optimize traffic load ############################################## 

def optimize_latency(Bw, h):
    # create a model
    m = Model('latency')
    # create decision variables
    x = m.addVars(an, an, vtype = GRB.BINARY, name = 'x')
    y = m.addVars(an, dc, vtype = GRB.BINARY, name = 'y')
    
    ######### add constraints ##########
    # x is symmetric
    m.addConstrs((x[i,j] == x[j,i] for i in an for j in an), 'symmetric constraint')
    # Two service areas cant use the same stateful functions when x[i,j] = 0
    m.addConstrs((y[i,t] + y[j,t] <= x[i,j] + 1 for i in an for j in an for t in dc), 'x = 0 constraint')
    # Two service areas use the same stateful functions when x[i,j] = 1
    m.addConstrs(((y[i,t] - y[j,t] <= 1 - x[i,j]) for i in an for j in an for t in dc), 'x = 1 constraint')
    # One access node connects to only one dc
    m.addConstrs((sum(y[i,t] for t in dc) == 1 for i in an),'one dc for access node')
    # high availability constraint
    m.addConstr(sum((1-x[i,j]) for i in an for j in an if i != j) >= 1,'ha')
    # maximum state transfer frequency
    m.addConstr(sum(h*(1 - x[i,j]) for i in an for j in an if i != j) <= State_max, 'max state transfer constraint')
    
    ######## Objective function #########
    # Optimize latency budget
    m.setObjective(sum(y[i,t]*C[i,t]*(Sd/Bw + Lw + Wg) for i in an for t in dc),  GRB.MINIMIZE)
        
    
    m.optimize()
    
    ######## Calculate performance metrics #########
    # latency
    latency = sum(C[i,t]*(Sd/Bw + Lw + Wg)*getattr(y[i,t], 'X') for i in an for t in dc) 
    
    # state transfer frequency
    state_transfer = sum(h*(1-getattr(x[i,j], 'X')) for i in an for j in an if i != j)
    
    # number of function
    num_func= M_dc - sum(prod(1 - getattr(y[i,t], 'X') for i in an) for t in dc)
    
    # total cost
    cost = latency + state_transfer
    
    # total metrics
    total_metrics = [Bw, h, state_transfer, latency, num_func, cost]
    
    # utopia point, nadir point
    worst_state =  state_transfer
    best_latency = latency

    return total_metrics, worst_state, best_latency


################# Trade off solution ##########################################
    
def optimize_pareto(Bw, h, ju_s, ju_l, jn_s, jn_l):
    # create a model
    m = Model('Pareto')
    # create decision variables
    x = m.addVars(an, an, vtype = GRB.BINARY, name = 'x')
    y = m.addVars(an, dc, vtype = GRB.BINARY, name = 'y')
    
    ######### add constraints ##########
    m.addConstrs((x[i,j] == x[j,i] for i in an for j in an), 'symmetric constraint')
    # Two service areas cant use the same stateful functions when x[i,j] = 0
    m.addConstrs((y[i,t] + y[j,t] <= x[i,j] + 1 for i in an for j in an for t in dc), 'x = 0 constraint')
    # Two service areas use the same stateful functions when x[i,j] = 1
    m.addConstrs(((y[i,t] - y[j,t] <= 1 - x[i,j]) for i in an for j in an for t in dc), 'x = 1 constraint')
    # Each should be at least managed by one dc
    m.addConstrs((sum(y[i,t] for t in dc) == 1 for i in an),'one dc for access node')
    # high availability constraint
    m.addConstr(sum((1-x[i,j]) for i in an for j in an if i != j) >= 1,'ha')
    # maximum state transfer frequency
    m.addConstr(sum(h*(1 - x[i,j]) for i in an for j in an if i != j) <=  jn_s - 1, 'max state transfer constraint')
    # maximum latency constraint
    m.addConstr(sum(y[i,t]*C[i,t]*(Sd/Bw + Lw + Wg) for i in an for t in dc) <= jn_l, 'latency maximum constraint')
    
    ######## Objective function #########
    m.setObjective((sum(y[i,t]*C[i,t]*(Sd/Bw + Lw + Wg) for i in an for t in dc)-ju_l)/(jn_l - ju_l) + 
                   (sum(h*(1 - x[i,j]) for i in an for j in an if i != j)-ju_s)/(jn_s-ju_s), GRB.MINIMIZE)
    
    m.optimize()
    
    ######## Calculate performance metrics #########
    # latency
    latency = sum(C[i,t]*getattr(y[i,t], 'X')*(Sd/Bw + Lw + Wg) for i in an for t in dc)
    
    # state transfer frequency
    state_transfer = sum(h*(1-getattr(x[i,j], 'X')) for i in an for j in an if i != j)
    
    # total cost
    cost = latency + state_transfer
    
    # number of functions
    num_func= M_dc - sum(prod(1 - getattr(y[i,t], 'X') for i in an) for t in dc)
    
    # total metrics
    total_metrics = [Bw, h, state_transfer, latency, num_func, cost]
    
    
    return total_metrics


################# obtain results ############################################## 
handover_list = [0.5] + [5*x for x in range(20) if x!=0]
bandwidth_list =  [10*x for x in range(20) if x!=0] + [200]
#h_ = 50
Bw_ = 200
idex = 0
for h_ in handover_list:
    metric_1, best_state, worst_latency = optimize_state_transfer(Bw_, h_)
    print(Bw_,'test1')
    metric_2, worst_state, best_latency = optimize_latency(Bw_, h_)
    print(Bw_,'test2')
    print( best_state, worst_state, best_latency, worst_latency)
    metric_3 = optimize_pareto(Bw_, h_, best_state, best_latency, worst_state, worst_latency)
    print(Bw_,'test3')
    
    df_state.iloc[idex] = (Bw_, h_, metric_1[2], metric_2[2], metric_3[2])
    df_latency.iloc[idex] = (Bw_, h_, metric_1[3], metric_2[3], metric_3[3])
    df_num_func.iloc[idex] = (Bw_, h_, metric_1[4], metric_2[4],  metric_3[4])
    cost_ost = (metric_1[2] - best_state)/(worst_state-best_state) + (metric_1[3] - best_latency)/(worst_latency - best_latency)
    cost_opl = (metric_2[2] - best_state)/(worst_state-best_state) + (metric_2[3] - best_latency)/(worst_latency - best_latency)
    cost_po = (metric_3[2] - best_state)/(worst_state-best_state) + (metric_3[3] - best_latency)/(worst_latency - best_latency)
    df_cost.iloc[idex] = (Bw_, h_, cost_ost, cost_opl, cost_po)
    idex = idex + 1
    

    
################# make figures ################################################
plt.figure()
state_plot = df_state.plot(x = 'Handover', y = ['OST', 'OPL', 'PO'],  style = ['r*-', 'go-', 'b^-'])
state_plot.set(ylabel = 'State Transfer Frequency')
load_plot = df_latency.plot(x = 'Handover', y = ['OST', 'OPL', 'PO'],  style = ['r*-', 'go-', 'b^-'])
load_plot.set(ylabel = 'Total Packet Latency')
cost_plot = df_cost.plot(x = 'Handover', y = ['OST', 'OPL', 'PO'], style = ['r*-', 'go-', 'b^-'])
cost_plot.set(ylabel = 'Total cost')
num_func_plot = df_num_func.plot(x = 'Handover', y = ['OST', 'OPL', 'PO'], style = ['r*-', 'go-', 'b^-'])
num_func_plot.set(ylabel = 'Number of Functions')

#state_plot = df_state.plot(x = 'Bandwidth', y = ['OST', 'OPL', 'PO'],  style = ['r*-', 'go-', 'b^-'])
#state_plot.set(ylabel = 'State Transfer Frequency')
#load_plot = df_latency.plot(x = 'Bandwidth', y = ['OST', 'OPL', 'PO'],  style = ['r*-', 'go-', 'b^-'])
#load_plot.set(ylabel = 'Total Packet Latency')
#cost_plot = df_cost.plot(x = 'Bandwidth', y = ['OST', 'OPL', 'PO'], style = ['r*-', 'go-', 'b^-'])
#cost_plot.set(ylabel = 'Total cost')
#num_func_plot = df_num_func.plot(x = 'Bandwidth', y = ['OST', 'OPL', 'PO'], style = ['r*-', 'go-', 'b^-'])
#num_func_plot.set(ylabel = 'Number of Functions')


