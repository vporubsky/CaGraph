# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 23:10:49 2020

@author: Veronica Porubsky

Title: get context-specific connectivity
"""
from dg_network_graph import DGNetworkGraph as nng
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from cycler import cycler
sns.set(style="whitegrid")

def get_context_active_indices(con_act):
    np.genfromtxt(con_act, delimiter=',')
    nonspecific_indices  = []
    con_A_active_indices = []
    con_B_active_indices = []
    for i in range(len(con_act)):
        if con_act[i] == 0:
            nonspecific_indices.append(i)
        elif con_act[i] == 1:
            con_A_active_indices.append(i)
        elif con_act[i] == 2:
            con_B_active_indices.append(i)
    return nonspecific_indices, con_A_active_indices, con_B_active_indices

path_to_data = '/LC-DG-FC-data/'
path_to_export = '/scratch_files/General_Exam/'

#%% Day 1 analyses
day1_untreated_ids = ['1055-1_D1',  '1055-2_D1','1055-3_D1', '1055-4_D1', '14-0_D1']

con_act_substring = '_neuron_context_active.csv'
calcium_traces_substring = '_smoothed_calcium_traces.csv'

D1_both_con_A_active_con_A_connectivity = [] # two context A active cells correlation in context A
D1_both_con_A_active_con_B_connectivity = [] # two context A active cells correlation in context B
D1_both_con_B_active_con_A_connectivity = [] # two context B active cells correlation in context A
D1_both_con_B_active_con_B_connectivity = [] # two context B active cells correlation in context B
D1_both_ns_con_A_connectivity = [] # two nonspecific cells correlation in context A
D1_both_ns_con_B_connectivity = [] # two nonspecific cells correlation in context B
D1_con_A_con_B_active_con_A_connectivity = [] # a context A and context B active cell correlation in context A
D1_con_A_con_B_active_con_B_connectivity = [] # a context A and context B active cell correlation in context B
D1_ns_con_A_active_con_A_connectivity = [] # a nonspecific cell and context A active cell correlation in context A
D1_ns_con_A_active_con_B_connectivity = [] # a nonspecific cell and context A active cell correlation in context B
D1_ns_con_B_active_con_A_connectivity = [] # a nonspecific cell and context B active cell correlation in context A
D1_ns_con_B_active_con_B_connectivity = [] # a nonspecific cell and context B active cell correlation in context B
D1_ns_ns_con_A_connectivity = [] # a nonspecific cell and nonspecific cell correlation in context A
D1_ns_ns_con_B_connectivity = [] # a nonspecific cell and nonspecific cell correlation in context B

ns_idx_list = []
A_idx_list = []
B_idx_list =[]

for mouse_id in day1_untreated_ids:

    con_act = np.genfromtxt(path_to_data + mouse_id + con_act_substring, delimiter=',')
    ns_idx, A_idx, B_idx = get_context_active_indices(path_to_data + mouse_id + con_act_substring) # sort indices of context active cells
    nn = nng(path_to_data + mouse_id + calcium_traces_substring)
    pearsons_con_A = nn.con_A_pearsons_correlation_matrix
    pearsons_con_B = nn.con_B_pearsons_correlation_matrix

    for neuron1 in range(len(con_act)): 
        for neuron2 in range(len(con_act)):   
            if not neuron1 == neuron2:
                if neuron1 and neuron2 in A_idx:
                    D1_both_con_A_active_con_A_connectivity.append(pearsons_con_A[neuron1,neuron2])
                    D1_both_con_A_active_con_B_connectivity.append(pearsons_con_B[neuron1,neuron2])
                elif neuron1 and neuron2 in B_idx:
                    D1_both_con_B_active_con_A_connectivity.append(pearsons_con_A[neuron1,neuron2])
                    D1_both_con_B_active_con_B_connectivity.append(pearsons_con_B[neuron1,neuron2])
                elif neuron1 and neuron2 in ns_idx:
                    D1_both_ns_con_A_connectivity.append(pearsons_con_A[neuron1,neuron2])
                    D1_both_ns_con_B_connectivity.append(pearsons_con_B[neuron1,neuron2])
                elif (neuron1 in A_idx and neuron2 in B_idx) or (neuron2 in A_idx and neuron1 in B_idx):
                    D1_con_A_con_B_active_con_A_connectivity.append(pearsons_con_A[neuron1,neuron2])
                    D1_con_A_con_B_active_con_B_connectivity.append(pearsons_con_B[neuron1,neuron2])
                elif (neuron1 in A_idx and neuron2 in ns_idx) or (neuron2 in A_idx and neuron1 in ns_idx):
                    D1_ns_con_A_active_con_A_connectivity.append(pearsons_con_A[neuron1,neuron2])
                    D1_ns_con_A_active_con_B_connectivity.append(pearsons_con_B[neuron1,neuron2])
                elif (neuron1 in B_idx and neuron2 in ns_idx) or (neuron2 in B_idx and neuron1 in ns_idx):
                    D1_ns_con_B_active_con_A_connectivity.append(pearsons_con_A[neuron1,neuron2])
                    D1_ns_con_B_active_con_B_connectivity.append(pearsons_con_B[neuron1,neuron2])
                else:
                    D1_ns_ns_con_A_connectivity.append(pearsons_con_A[neuron1,neuron2])
                    D1_ns_ns_con_B_connectivity.append(pearsons_con_B[neuron1,neuron2])
print('Day 1:')               
print('Correlation between two context A active cells in context A: {:.3f} mean, {:.3f} stdev'.format(np.mean(D1_both_con_A_active_con_A_connectivity), np.std(D1_both_con_A_active_con_A_connectivity)))        
print('Correlation between two context A active cells in context B: {:.3f} mean, {:.3f} stdev'.format(np.mean(D1_both_con_A_active_con_B_connectivity), np.std(D1_both_con_A_active_con_B_connectivity)))  
print('Correlation between two context B active cells in context A: {:.3f} mean, {:.3f} stdev'.format(np.mean(D1_both_con_B_active_con_A_connectivity), np.std(D1_both_con_B_active_con_A_connectivity)))        
print('Correlation between two context B active cells in context B: {:.3f} mean, {:.3f} stdev'.format(np.mean(D1_both_con_B_active_con_B_connectivity), np.std(D1_both_con_B_active_con_B_connectivity)))  
print('Correlation between a context A and context B active cell in context A: {:.3f} mean, {:.3f} stdev'.format(np.mean(D1_con_A_con_B_active_con_A_connectivity), np.std(D1_con_A_con_B_active_con_B_connectivity)))        
print('Correlation between a context A and context B active cell in context B: {:.3f} mean, {:.3f} stdev'.format(np.mean(D1_con_A_con_B_active_con_B_connectivity), np.std(D1_con_A_con_B_active_con_B_connectivity)))  
print('Correlation between a nonspecific and context A active cell in context A: {:.3f} mean, {:.3f} stdev'.format(np.mean(D1_ns_con_A_active_con_A_connectivity), np.std(D1_ns_con_A_active_con_A_connectivity)))        
print('Correlation between a nonspecific and context A active cell in context B: {:.3f} mean, {:.3f} stdev'.format(np.mean(D1_ns_con_A_active_con_B_connectivity), np.std(D1_ns_con_A_active_con_B_connectivity)))  
print('Correlation between a nonspecific and context B active cell in context A: {:.3f} mean, {:.3f} stdev'.format(np.mean(D1_ns_con_B_active_con_A_connectivity), np.std(D1_ns_con_B_active_con_A_connectivity)))        
print('Correlation between a nonspecific and context B active cell in context B: {:.3f} mean, {:.3f} stdev'.format(np.mean(D1_ns_con_B_active_con_B_connectivity), np.std(D1_ns_con_B_active_con_B_connectivity)))  
print('Correlation between a nonspecific and nonspecific cell in context A: {:.3f} mean, {:.3f} stdev'.format(np.mean(D1_ns_ns_con_A_connectivity), np.std(D1_ns_ns_con_A_connectivity)))        
print('Correlation between a nonspecific and nonspecific cell in context B: {:.3f} mean, {:.3f} stdev'.format(np.mean(D1_ns_ns_con_B_connectivity), np.std(D1_ns_ns_con_B_connectivity)))  

print('Mean percentage of context A active cells: {:.3f}%'.format(np.mean(A_idx_list)*100))
print('Mean percentage of context B active cells: {:.3f}%'.format(np.mean(B_idx_list)*100))
print('Mean percentage of nonspecific cells: {:.3f}%'.format(np.mean(ns_idx_list)*100))
                    
#%% Day 9 analyses
day9_untreated_ids = ['1055-1_D9',  '1055-2_D9','1055-3_D9', '1055-4_D9', '14-0_D9']

con_act_substring = '_neuron_context_active.csv'
calcium_traces_substring = '_smoothed_calcium_traces.csv'

D9_both_con_A_active_con_A_connectivity = [] # two context A active cells correlation in context A
D9_both_con_A_active_con_B_connectivity = [] # two context A active cells correlation in context B
D9_both_con_B_active_con_A_connectivity = [] # two context B active cells correlation in context A
D9_both_con_B_active_con_B_connectivity = [] # two context B active cells correlation in context B
D9_both_ns_con_A_connectivity = [] # two nonspecific cells correlation in context A
D9_both_ns_con_B_connectivity = [] # two nonspecific cells correlation in context B
D9_con_A_con_B_active_con_A_connectivity = [] # a context A and context B active cell correlation in context A
D9_con_A_con_B_active_con_B_connectivity = [] # a context A and context B active cell correlation in context B
D9_ns_con_A_active_con_A_connectivity = [] # a nonspecific cell and context A active cell correlation in context A
D9_ns_con_A_active_con_B_connectivity = [] # a nonspecific cell and context A active cell correlation in context B
D9_ns_con_B_active_con_A_connectivity = [] # a nonspecific cell and context B active cell correlation in context A
D9_ns_con_B_active_con_B_connectivity = [] # a nonspecific cell and context B active cell correlation in context B
D9_ns_ns_con_A_connectivity = [] # a nonspecific cell and nonspecific cell correlation in context A
D9_ns_ns_con_B_connectivity = [] # a nonspecific cell and nonspecific cell correlation in context B


ns_idx_list = []
A_idx_list = []
B_idx_list =[]

for mouse_id in day9_untreated_ids:
    
    con_act = path_to_data + mouse_id + con_act_substring
    
    ns_idx, A_idx, B_idx = get_context_active_indices(path_to_data + mouse_id + con_act_substring) # sort indices of context active cells
    ns_idx_list.append(len(ns_idx)/len(con_act))
    A_idx_list.append(len(A_idx)/len(con_act))
    B_idx_list.append(len(B_idx)/len(con_act))
    
    nn = nng(path_to_data + mouse_id + calcium_traces_substring)
    pearsons_con_A = nn.con_A_pearsons_correlation_matrix
    pearsons_con_B = nn.con_B_pearsons_correlation_matrix

    for neuron1 in range(len(con_act)): 
        for neuron2 in range(len(con_act)):   
            if not neuron1 == neuron2:
                if neuron1 and neuron2 in A_idx:
                    D9_both_con_A_active_con_A_connectivity.append(pearsons_con_A[neuron1,neuron2])
                    D9_both_con_A_active_con_B_connectivity.append(pearsons_con_B[neuron1,neuron2])
                elif neuron1 and neuron2 in B_idx:
                    D9_both_con_B_active_con_A_connectivity.append(pearsons_con_A[neuron1,neuron2])
                    D9_both_con_B_active_con_B_connectivity.append(pearsons_con_B[neuron1,neuron2])
                elif neuron1 and neuron2 in ns_idx:
                    D9_both_ns_con_A_connectivity.append(pearsons_con_A[neuron1,neuron2])
                    D9_both_ns_con_B_connectivity.append(pearsons_con_B[neuron1,neuron2])
                elif (neuron1 in A_idx and neuron2 in B_idx) or (neuron2 in A_idx and neuron1 in B_idx):
                    D9_con_A_con_B_active_con_A_connectivity.append(pearsons_con_A[neuron1,neuron2])
                    D9_con_A_con_B_active_con_B_connectivity.append(pearsons_con_B[neuron1,neuron2])
                elif (neuron1 in A_idx and neuron2 in ns_idx) or (neuron2 in A_idx and neuron1 in ns_idx):
                    D9_ns_con_A_active_con_A_connectivity.append(pearsons_con_A[neuron1,neuron2])
                    D9_ns_con_A_active_con_B_connectivity.append(pearsons_con_B[neuron1,neuron2])
                elif (neuron1 in B_idx and neuron2 in ns_idx) or (neuron2 in B_idx and neuron1 in ns_idx):
                    D9_ns_con_B_active_con_A_connectivity.append(pearsons_con_A[neuron1,neuron2])
                    D9_ns_con_B_active_con_B_connectivity.append(pearsons_con_B[neuron1,neuron2])
                else:
                    D9_ns_ns_con_A_connectivity.append(pearsons_con_A[neuron1,neuron2])
                    D9_ns_ns_con_B_connectivity.append(pearsons_con_B[neuron1,neuron2])
print('Day 9:')
print('Correlation between two context A active cells in context A: {:.3f} mean, {:.3f} stdev'.format(np.mean(D9_both_con_A_active_con_A_connectivity), np.std(D9_both_con_A_active_con_A_connectivity)))        
print('Correlation between two context A active cells in context B: {:.3f} mean, {:.3f} stdev'.format(np.mean(D9_both_con_A_active_con_B_connectivity), np.std(D9_both_con_A_active_con_B_connectivity)))  
print('Correlation between two context B active cells in context A: {:.3f} mean, {:.3f} stdev'.format(np.mean(D9_both_con_B_active_con_A_connectivity), np.std(D9_both_con_B_active_con_A_connectivity)))        
print('Correlation between two context B active cells in context B: {:.3f} mean, {:.3f} stdev'.format(np.mean(D9_both_con_B_active_con_B_connectivity), np.std(D9_both_con_B_active_con_B_connectivity)))  
print('Correlation between a context A and context B active cell in context A: {:.3f} mean, {:.3f} stdev'.format(np.mean(D9_con_A_con_B_active_con_A_connectivity), np.std(D9_con_A_con_B_active_con_B_connectivity)))        
print('Correlation between a context A and context B active cell in context B: {:.3f} mean, {:.3f} stdev'.format(np.mean(D9_con_A_con_B_active_con_B_connectivity), np.std(D9_con_A_con_B_active_con_B_connectivity)))  
print('Correlation between a nonspecific and context A active cell in context A: {:.3f} mean, {:.3f} stdev'.format(np.mean(D9_ns_con_A_active_con_A_connectivity), np.std(D9_ns_con_A_active_con_A_connectivity)))        
print('Correlation between a nonspecific and context A active cell in context B: {:.3f} mean, {:.3f} stdev'.format(np.mean(D9_ns_con_A_active_con_B_connectivity), np.std(D9_ns_con_A_active_con_B_connectivity)))  
print('Correlation between a nonspecific and context B active cell in context A: {:.3f} mean, {:.3f} stdev'.format(np.mean(D9_ns_con_B_active_con_A_connectivity), np.std(D9_ns_con_B_active_con_A_connectivity)))        
print('Correlation between a nonspecific and context B active cell in context B: {:.3f} mean, {:.3f} stdev'.format(np.mean(D9_ns_con_B_active_con_B_connectivity), np.std(D9_ns_con_B_active_con_B_connectivity)))  
print('Correlation between a nonspecific and nonspecific cell in context A: {:.3f} mean, {:.3f} stdev'.format(np.mean(D9_ns_ns_con_A_connectivity), np.std(D9_ns_ns_con_A_connectivity)))        
print('Correlation between a nonspecific and nonspecific cell in context B: {:.3f} mean, {:.3f} stdev'.format(np.mean(D9_ns_ns_con_B_connectivity), np.std(D9_ns_ns_con_B_connectivity)))  

print('Mean percentage of context A active cells: {:.3f}%'.format(np.mean(A_idx_list)*100))
print('Mean percentage of context B active cells: {:.3f}%'.format(np.mean(B_idx_list)*100))
print('Mean percentage of nonspecific cells: {:.3f}%'.format(np.mean(ns_idx_list)*100))

#%% Plot context A active connectivity metrics -- in context A
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.rc('axes', prop_cycle=(cycler('color', ["salmon", "darkturquoise", "salmon", "darkturquoise", "salmon", "darkturquoise"])))
labels = ['D1_A_A_con_A', 'D9_A_A_con_A', 'D1_A_B_con_A', 'D9_A_B_con_A', 'D1_A_NS_con_A', 'D9_A_NS_con_A']
raw = [D1_both_con_A_active_con_A_connectivity, D9_both_con_A_active_con_A_connectivity,\
       D1_con_A_con_B_active_con_A_connectivity, D9_con_A_con_B_active_con_A_connectivity,\
       D1_ns_con_A_active_con_A_connectivity, D9_ns_con_A_active_con_A_connectivity]
plt.title('Context A Active -- Correlation in Context A')
plt.xticks([])
sns.violinplot(data=raw, whis = 1.5);
plt.savefig('CSE528_conA_active_conA_correlations.png', dpi = 300)

#%% Plot context A active connectivity metrics -- in context B
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.rc('axes', prop_cycle=(cycler('color', ["salmon", "darkturquoise", "salmon", "darkturquoise", "salmon", "darkturquoise"])))
labels = ['D1_A_A_con_B', 'D9_A_A_con_B', 'D1_A_B_con_B', 'D9_A_B_con_B', 'D1_A_NS_con_B', 'D9_A_NS_con_B']
raw = [D1_both_con_A_active_con_B_connectivity, D9_both_con_A_active_con_B_connectivity,\
       D1_con_A_con_B_active_con_B_connectivity, D9_con_A_con_B_active_con_B_connectivity,\
       D1_ns_con_A_active_con_B_connectivity, D9_ns_con_A_active_con_B_connectivity]
plt.title('Context A Active -- Correlation in Context B')
plt.xticks([])
sns.violinplot(data=raw, whis = 1.5);
plt.savefig(path_to_export + 'CSE528_conA_active_conB_correlations.png', dpi = 300)

#%% Plot context B active connectivity metrics
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.rc('axes', prop_cycle=(cycler('color', ["salmon", "darkturquoise", "salmon", "darkturquoise", "salmon", "darkturquoise"])))
labels = ['D1_B_B_con_A', 'D9_B_B_con_A', 'D1_A_B_con_A', 'D9_A_B_con_A', 'D1_B_NS_con_A', 'D9_B_NS_con_A']
raw = [D1_both_con_B_active_con_A_connectivity, D9_both_con_B_active_con_A_connectivity,\
       D1_con_A_con_B_active_con_A_connectivity, D9_con_A_con_B_active_con_A_connectivity,\
       D1_ns_con_B_active_con_A_connectivity, D9_ns_con_B_active_con_A_connectivity]
plt.title('Context B Active -- Correlation in Context A')
plt.xticks([])
sns.violinplot(data=raw, whis = 1.5);
plt.savefig(path_to_export + 'CSE528_conB_active_conA_correlations.png', dpi = 300)

#%% Plot context B active connectivity metrics -- in context B
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.rc('axes', prop_cycle=(cycler('color', ["salmon", "darkturquoise", "salmon", "darkturquoise", "salmon", "darkturquoise"])))
labels = ['D1_B_B_con_B', 'D9_B_B_con_B', 'D1_A_B_con_B', 'D9_A_B_con_B', 'D1_B_NS_con_B', 'D9_B_NS_con_B']
raw = [D1_both_con_B_active_con_B_connectivity, D9_both_con_B_active_con_B_connectivity,\
       D1_con_A_con_B_active_con_B_connectivity, D9_con_A_con_B_active_con_B_connectivity,\
       D1_ns_con_B_active_con_B_connectivity, D9_ns_con_B_active_con_B_connectivity]
plt.title('Context B Active -- Correlation in Context B')
plt.xticks([])
sns.violinplot(data=raw, whis = 1.5);
plt.savefig(path_to_export + 'CSE528_conB_active_conB_correlations.png', dpi = 300)
