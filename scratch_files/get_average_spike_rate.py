# -*- coding: utf-8 -*-
"""
Created on Sun Jun 7 17:05:41 2020

@author: Veronica Porubsky

Title: get firing average firing rates
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from cycler import cycler
sns.set(style="whitegrid")

def get_average_firing_rate(spike_train, bin_size, sampling_rate):
    fr_avg = np.zeros(int(len(spike_train)/bin_size))
    idx = 0
    end_idx = bin_size
    for i in range(int(len(spike_train)/bin_size)):
        for j in range(bin_size):
            if spike_train[idx]:
                fr_avg[i] += 1
            idx += 1
        end_idx += bin_size
    return np.average(fr_avg/(sampling_rate*bin_size))

def get_context_active_indices(con_act):
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

# open a file, where you stored the pickled data
file = open('348-1_D9_spike_train.pkl', 'rb')
# dump information to that file
data = pickle.load(file)
# close the file
file.close()

avg_fr_con_A = get_average_firing_rate(data[0,1800:3600], bin_size = 75, sampling_rate = 0.1)
avg_fr_con_B = get_average_firing_rate(data[0,1800:3600], bin_size = 75, sampling_rate = 0.1)

# open a file, where you stored the pickled data
file = open('348-1_D9_spike_times.pkl', 'rb')
# dump information to that file
spike_times = pickle.load(file)
# close the file
file.close()
plt.eventplot(spike_times)

#%% 
day1_untreated_ids = ['1055-1_D1',  '1055-2_D1','1055-3_D1', '1055-4_D1', '14-0_D1']

sr = 0.1
con_act_substring = '_neuron_context_active.npy'
spike_train_substring = '_spike_train.pkl'

D1_con_A_active_con_A_average_firing_rates = []
D1_con_A_active_con_B_average_firing_rates = []
D1_con_B_active_con_A_average_firing_rates = []
D1_con_B_active_con_B_average_firing_rates = []
D1_ns_con_A_average_firing_rates = []
D1_ns_con_B_average_firing_rates = []

D1_ns_idx_list = []
D1_A_idx_list = []
D1_B_idx_list =[]

for mouse_id in day1_untreated_ids: # change which id list to use for day 1 or day 9 results
    con_act = np.load(mouse_id + con_act_substring)[0] # load context active designations
    
    ns_idx, A_idx, B_idx = get_context_active_indices(con_act) # sort indices of context active cells
    D1_ns_idx_list.append(len(ns_idx)/len(con_act))
    D1_A_idx_list.append(len(A_idx)/len(con_act))
    D1_B_idx_list.append(len(B_idx)/len(con_act))
    
    file = open(mouse_id + spike_train_substring, 'rb')
    data = pickle.load(file)
    file.close()

    for neuron in range(len(data)): 
        avg_fr_con_A = get_average_firing_rate(data[neuron, :1800], bin_size = 75, sampling_rate = sr)
        avg_fr_con_B = get_average_firing_rate(data[neuron, 1800:3600], bin_size = 75, sampling_rate = sr)
        if neuron in A_idx:
            D1_con_A_active_con_A_average_firing_rates.append(avg_fr_con_A)
            D1_con_A_active_con_B_average_firing_rates.append(avg_fr_con_B)
        elif neuron in B_idx:
            D1_con_B_active_con_A_average_firing_rates.append(avg_fr_con_A)
            D1_con_B_active_con_B_average_firing_rates.append(avg_fr_con_B)
        else:
            D1_ns_con_A_average_firing_rates.append(avg_fr_con_A)
            D1_ns_con_B_average_firing_rates.append(avg_fr_con_B)

#%% Print report
print('Context A active cells in context A: {:.3f} mean, {:.3f} stdev'.format(np.mean(D1_con_A_active_con_A_average_firing_rates), np.std(D1_con_A_active_con_A_average_firing_rates)))        
print('Context A active cells in context B: {:.3f} mean, {:.3f} stdev'.format(np.mean(D1_con_A_active_con_B_average_firing_rates), np.std(D1_con_A_active_con_B_average_firing_rates)))  
print('Context B active cells in context A: {:.3f} mean, {:.3f} stdev'.format(np.mean(D1_con_B_active_con_A_average_firing_rates), np.std(D1_con_B_active_con_A_average_firing_rates)))  
print('Context B active cells in context B: {:.3f} mean, {:.3f} stdev'.format(np.mean(D1_con_B_active_con_B_average_firing_rates), np.std(D1_con_B_active_con_B_average_firing_rates)))  
print('Nonspecific cells in context A: {:.3f} mean, {:.3f} stdev'.format(np.mean(D1_ns_con_A_average_firing_rates), np.std(D1_ns_con_A_average_firing_rates)))  
print('Nonspecific cells in context B: {:.3f} mean, {:.3f} stdev'.format(np.mean(D1_ns_con_B_average_firing_rates), np.std(D1_ns_con_B_average_firing_rates)))  

print('Mean percentage of context A active cells: {:.3f}%'.format(np.mean(D1_A_idx_list)*100))
print('Mean percentage of context B active cells: {:.3f}%'.format(np.mean(D1_B_idx_list)*100))
print('Mean percentage of nonspecific cells: {:.3f}%'.format(np.mean(D1_ns_idx_list)*100))

#%% 
day9_untreated_ids = ['1055-1_D9',  '1055-2_D9','1055-3_D9', '1055-4_D9'] # file does not exist: '14-0_D9'

sr = 0.1
con_act_substring = '_neuron_context_active.npy'
spike_train_substring = '_spike_train.pkl'

D9_con_A_active_con_A_average_firing_rates = []
D9_con_A_active_con_B_average_firing_rates = []
D9_con_B_active_con_A_average_firing_rates = []
D9_con_B_active_con_B_average_firing_rates = []
D9_ns_con_A_average_firing_rates = []
D9_ns_con_B_average_firing_rates = []

D9_ns_idx_list = []
D9_A_idx_list = []
D9_B_idx_list =[]

for mouse_id in day9_untreated_ids: # change which id list to use for day 1 or day 9 results
    con_act = np.load(mouse_id + con_act_substring)[0] # load context active designations
    
    ns_idx, A_idx, B_idx = get_context_active_indices(con_act) # sort indices of context active cells
    D9_ns_idx_list.append(len(ns_idx)/len(con_act))
    D9_A_idx_list.append(len(A_idx)/len(con_act))
    D9_B_idx_list.append(len(B_idx)/len(con_act))
    
    file = open(mouse_id + spike_train_substring, 'rb')
    data = pickle.load(file)
    file.close()

    for neuron in range(len(data)): 
        avg_fr_con_A = get_average_firing_rate(data[neuron, :1800], bin_size = 75, sampling_rate = sr)
        avg_fr_con_B = get_average_firing_rate(data[neuron, 1800:3600], bin_size = 75, sampling_rate = sr)
        if neuron in A_idx:
            D9_con_A_active_con_A_average_firing_rates.append(avg_fr_con_A)
            D9_con_A_active_con_B_average_firing_rates.append(avg_fr_con_B)
        elif neuron in B_idx:
            D9_con_B_active_con_A_average_firing_rates.append(avg_fr_con_A)
            D9_con_B_active_con_B_average_firing_rates.append(avg_fr_con_B)
        else:
            D9_ns_con_A_average_firing_rates.append(avg_fr_con_A)
            D9_ns_con_B_average_firing_rates.append(avg_fr_con_B)

#%% Print report
print('Context A active cells in context A: {:.3f} mean, {:.3f} stdev'.format(np.mean(D9_con_A_active_con_A_average_firing_rates), np.std(D9_con_A_active_con_A_average_firing_rates)))        
print('Context A active cells in context B: {:.3f} mean, {:.3f} stdev'.format(np.mean(D9_con_A_active_con_B_average_firing_rates), np.std(D9_con_A_active_con_B_average_firing_rates)))  
print('Context B active cells in context A: {:.3f} mean, {:.3f} stdev'.format(np.mean(D9_con_B_active_con_A_average_firing_rates), np.std(D9_con_B_active_con_A_average_firing_rates)))  
print('Context B active cells in context B: {:.3f} mean, {:.3f} stdev'.format(np.mean(D9_con_B_active_con_B_average_firing_rates), np.std(D9_con_B_active_con_B_average_firing_rates)))  
print('Nonspecific cells in context A: {:.3f} mean, {:.3f} stdev'.format(np.mean(D9_ns_con_A_average_firing_rates), np.std(D9_ns_con_A_average_firing_rates)))  
print('Nonspecific cells in context B: {:.3f} mean, {:.3f} stdev'.format(np.mean(D9_ns_con_B_average_firing_rates), np.std(D9_ns_con_B_average_firing_rates)))  

print('Mean percentage of context A active cells: {:.3f}%'.format(np.mean(D9_A_idx_list)*100))
print('Mean percentage of context B active cells: {:.3f}%'.format(np.mean(D9_B_idx_list)*100))
print('Mean percentage of nonspecific cells: {:.3f}%'.format(np.mean(D9_ns_idx_list)*100))

#%% Plot context A active firing rates
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.rc('axes', prop_cycle=(cycler('color', ["darkturquoise", "darkturquoise","salmon", "salmon"])))

raw = [D1_con_A_active_con_A_average_firing_rates, D9_con_A_active_con_A_average_firing_rates,\
       D1_con_A_active_con_B_average_firing_rates, D9_con_A_active_con_B_average_firing_rates]
plt.title('Context A Active Firing Rates')
plt.xticks([])
sns.violinplot(data=raw, whis = 1.5);
plt.savefig('CSE528_conA_active_firing_rates.png', dpi = 300)

#%% Plot context B active firing rates
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.rc('axes', prop_cycle=(cycler('color', ["darkturquoise", "darkturquoise","salmon", "salmon"])))

raw = [D1_con_B_active_con_A_average_firing_rates, D9_con_B_active_con_A_average_firing_rates,\
       D1_con_B_active_con_B_average_firing_rates, D9_con_B_active_con_B_average_firing_rates]
plt.title('Context B Active Firing Rates')
plt.xticks([])
sns.violinplot(data=raw, whis = 1.5);
plt.savefig('CSE528_conB_active_firing_rates.png', dpi = 300)

#%% Plot nonspecific firing rates
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.rc('axes', prop_cycle=(cycler('color', ["darkturquoise", "darkturquoise","salmon", "salmon"])))

raw = [D1_ns_con_A_average_firing_rates, D9_ns_con_A_average_firing_rates,\
       D1_ns_con_B_average_firing_rates, D9_ns_con_B_average_firing_rates]
plt.title('Nonspecific Firing Rates')
plt.xticks([])
sns.violinplot(data=raw, whis = 1.5);
plt.savefig('CSE528_ns_firing_rates.png', dpi = 300)