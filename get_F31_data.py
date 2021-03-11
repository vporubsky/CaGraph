# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:30:42 2020

@author: Veronica Porubsky

Title: Generate preliminary data for NIH F31 April 23, 2020 submission
"""
import networkx as nx
from neuronal_network_graph import neuronal_network_graph as nng
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import scipy
import seaborn as sns
sns.set(style="whitegrid")
my_pal = {"D1_A": "salmon", "D1_B": "darkturquoise", "D9_A":"salmon", "D9_B":"darkturquoise"}

#%% Load untreated data files - WT
day1_untreated = ['1055-1_D1_all_calcium_traces.npy',  '1055-2_D1_all_calcium_traces.npy','1055-3_D1_all_calcium_traces.npy', '1055-4_D1_all_calcium_traces.npy', '14-0_D1_all_calcium_traces.npy']
day9_untreated = ['1055-1_D9_all_calcium_traces.npy',  '1055-2_D9_all_calcium_traces.npy','1055-3_D9_all_calcium_traces.npy', '1055-4_D9_all_calcium_traces.npy', '14-0_D9_all_calcium_traces.npy']

all_files = [day1_untreated, day9_untreated]

#%% All measurements, separating contexts 
threshold = 0.3
names = []
data_mat = []

con_A_subnetworks_D1 = []
con_B_subnetworks_D1 = []
con_A_num_subnetworks_D1 = []
con_B_num_subnetworks_D1 = []
con_A_size_subnetworks_D1 = []
con_B_size_subnetworks_D1 = []

con_A_subnetworks_D9 = []
con_B_subnetworks_D9 = []
con_A_num_subnetworks_D9 = []
con_B_num_subnetworks_D9 = []
con_A_size_subnetworks_D9 = []
con_B_size_subnetworks_D9 = [] 


con_A_hubs_D1 = []
con_B_hubs_D1 = []

con_A_hubs_D9 = []
con_B_hubs_D9 = []

con_A_num_hubs_D1 = []
con_B_num_hubs_D1 = []

con_A_num_hubs_D9 = []
con_B_num_hubs_D9 = []

edges_A_D1 = []
edges_B_D1 = []

edges_A_D9 = []
edges_B_D9 = []

mouse_id_indices = []


for treatment_group_index in [0,1]:
    for mouse_id_index in range(len(all_files[treatment_group_index])):
        filename = all_files[treatment_group_index][mouse_id_index]
        mouse_id = filename.strip('_all_calcium_traces.npy')
        
        nn = nng(filename)
        num_neurons = nn.num_neurons
        
        conA = nn.get_context_A_graph(threshold = threshold)
        conB = nn.get_context_B_graph(threshold = threshold)
        
        subnetwork_A = list(nx.connected_components(conA))
        subnetwork_B = list(nx.connected_components(conB))
        
        
        
        connected_subnetworks_A = []
        num_connected_subnetworks_A = 0
        len_connected_subnetworks_A = []
        
        connected_subnetworks_B = []
        num_connected_subnetworks_B = 0
        len_connected_subnetworks_B = []
        
        for k in range(len(subnetwork_A)):
            if len(subnetwork_A[k]) > 1:
                connected_subnetworks_A.append(list(map(int, subnetwork_A[k])))
                num_connected_subnetworks_A += 1
                len_connected_subnetworks_A.append(len(subnetwork_A[k]))
                
                
        for k in range(len(subnetwork_B)):
            if len(subnetwork_B[k]) > 1:
                connected_subnetworks_B.append(list(map(int, subnetwork_B[k]))) 
                num_connected_subnetworks_B += 1
                len_connected_subnetworks_B.append(len(subnetwork_B[k]))
                

        if treatment_group_index == 0:
            con_A_subnetworks_D1.append(connected_subnetworks_A)
            con_B_subnetworks_D1.append(connected_subnetworks_B)
            con_A_num_subnetworks_D1.append(num_connected_subnetworks_A/num_neurons)
            con_B_num_subnetworks_D1.append(num_connected_subnetworks_B/num_neurons)
            con_A_size_subnetworks_D1.append(np.average(len_connected_subnetworks_A)/num_neurons)
            con_B_size_subnetworks_D1.append(np.average(len_connected_subnetworks_B)/num_neurons)
            edges_A_D1.append(list(conA.edges()))
            edges_B_D1.append(list(conB.edges()))
            
        elif treatment_group_index == 1:
            con_A_subnetworks_D9.append(connected_subnetworks_A)
            con_B_subnetworks_D9.append(connected_subnetworks_B)
            con_A_num_subnetworks_D9.append(num_connected_subnetworks_A/num_neurons)
            con_B_num_subnetworks_D9.append(num_connected_subnetworks_B/num_neurons)
            con_A_size_subnetworks_D9.append(np.average(len_connected_subnetworks_A)/num_neurons)
            con_B_size_subnetworks_D9.append(np.average(len_connected_subnetworks_B)/num_neurons) 
            edges_A_D9.append(list(conA.edges()))
            edges_B_D9.append(list(conB.edges()))
        
        try:
            hits_A, authorities_A = nx.hits(conA, max_iter=500)
            hits_B, authorities_B = nx.hits(conB, max_iter=500)
        
            hub_hits_A = []
            hub_hits_B = []
            
            for k in hits_A.keys():
                if hits_A[k] > 0.1:
                    hub_hits_A.append(k)
            
            len_hubs_A = len(hub_hits_A)
                
            for k in hits_B.keys():
                if hits_B[k] > 0.05: # 0.05 chosen as
                    hub_hits_B.append(k)
            
            len_hubs_B = len(hub_hits_B)
                
                
        except Exception as e:
            print(e)
            hub_hits_A = 'NaN'
            hub_hits_B = 'NaN'
            len_hubs_A = 0
            len_hubs_B = 0
            
        if treatment_group_index == 0:
            mouse_id_indices.append(mouse_id.replace('_D1', ''))
            con_A_hubs_D1.append(hub_hits_A)
            con_B_hubs_D1.append(hub_hits_B)
            con_A_num_hubs_D1.append(len_hubs_A/num_neurons)
            con_B_num_hubs_D1.append(len_hubs_B/num_neurons)
                
            
        elif treatment_group_index == 1:
            con_A_hubs_D9.append(hub_hits_A)
            con_B_hubs_D9.append(hub_hits_B)
            con_A_num_hubs_D9.append(len_hubs_A/num_neurons)
            con_B_num_hubs_D9.append(len_hubs_B/num_neurons)
        
#%% plot hub metrics with seaborn
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

labels = ['D1_A', 'D1_B', 'D9_A', 'D9_B']
raw = [con_A_num_hubs_D1, con_B_num_hubs_D1, con_A_num_hubs_D9, con_B_num_hubs_D9]

data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels, index=mouse_id_indices)
sns.boxplot(data=data, whis = 1.5, palette=my_pal);
sns.swarmplot(data=data, color = 'k');

y, h, col = data['D9_B'].max() + 0.005, 0.005, 'k'
x1, x2 = 2, 3  # columns to match
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)
plt.text((x1+x2)*.5, y+h+0.01, "p = 0.024", ha='center', va='bottom', color=col)
plt.ylabel('Normalized hub count')
# plt.savefig('F31_hubs_fig.png', dpi = 300)

#%% TEst


sns.boxplot(data=data, palette=my_pal)


#%% Compute hub statistics 
D1_A_B = scipy.stats.ttest_rel(con_A_num_hubs_D1, con_B_num_hubs_D1)
D9_A_B = scipy.stats.ttest_rel(con_A_num_hubs_D9, con_B_num_hubs_D9)
A_D1_D9 = scipy.stats.ttest_rel(con_A_num_hubs_D1, con_A_num_hubs_D9)
B_D1_D9 = scipy.stats.ttest_rel(con_B_num_hubs_D1, con_B_num_hubs_D9)

test_stats = [D1_A_B, D9_A_B, A_D1_D9, B_D1_D9]
print(test_stats)

#%% plot subnetwork number metrics with seaborn 
labels = ['D1_A', 'D1_B', 'D9_A', 'D9_B']
raw = [con_A_num_subnetworks_D1, con_B_num_subnetworks_D1, con_A_num_subnetworks_D9, con_B_num_subnetworks_D9]

data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels, index=mouse_id_indices)

fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
sns.boxplot(data=data, whis = 1.5, palette=my_pal);
sns.swarmplot(data=data, color = 'k');
y, h, col = data['D1_A'].max() + 0.02, 0.01, 'k'
x1, x2 = 0, 1  # columns to match
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)

plt.text((x1+x2)*.5, y+h+0.005, "p = 0.0625", ha='center', va='bottom', color=col)
plt.ylabel('Normalized subnetwork count')
plt.savefig('F31_subnetwork_num_fig.png', dpi = 300)

#%% plot subnetwork size  metrics with seaborn 
labels = ['D1_A', 'D1_B', 'D9_A', 'D9_B']
raw = [con_A_size_subnetworks_D1, con_B_size_subnetworks_D1, con_A_size_subnetworks_D9, con_B_size_subnetworks_D9]

data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels, index=mouse_id_indices)

fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
sns.boxplot(data=data, whis = 1.5, palette=my_pal);
sns.swarmplot(data=data, color = 'k');
y, h, col = data['D9_A'].max() + 0.05, 0.02, 'k'
x1, x2 = 0, 2  # columns to match
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)
plt.text((x1+x2)*.5, y+h+0.075, "p = 0.0034", ha='center', va='bottom', color=col)
plt.ylabel('Normalized average subnetwork size')
plt.savefig('F31_subnetwork_size_fig.png', dpi = 300)

#%% Compute subnetwork number statistics 
D1_A_B = scipy.stats.ttest_rel(con_A_num_subnetworks_D1, con_B_num_subnetworks_D1)
D9_A_B = scipy.stats.ttest_rel(con_A_num_subnetworks_D9, con_B_num_subnetworks_D9)
A_D1_D9 = scipy.stats.ttest_rel(con_A_num_subnetworks_D1, con_A_num_subnetworks_D9)
B_D1_D9 = scipy.stats.ttest_rel(con_B_num_subnetworks_D1, con_B_num_subnetworks_D9)

test_stats = [D1_A_B, D9_A_B, A_D1_D9, B_D1_D9]
print(test_stats)

#%% Compute subnetwork size statistics 
D1_A_B = scipy.stats.ttest_rel(con_A_size_subnetworks_D1, con_B_size_subnetworks_D1)
D9_A_B = scipy.stats.ttest_rel(con_A_size_subnetworks_D9, con_B_size_subnetworks_D9)
A_D1_D9 = scipy.stats.ttest_rel(con_A_size_subnetworks_D1, con_A_size_subnetworks_D9)
B_D1_D9 = scipy.stats.ttest_rel(con_B_size_subnetworks_D1, con_B_size_subnetworks_D9)

test_stats = [D1_A_B, D9_A_B, A_D1_D9, B_D1_D9]
print(test_stats)

#%% Check if hubs are differentially identified - can only compare between contexts
percent_shared = []
for idx in range(0,5):
    y = con_A_hubs_D9[idx]
    x = con_B_hubs_D9[idx]
    if len(y) >= len(x) and len(y) != 0 and y != 'NaN':
        print([i in x for i in y])
        percent_shared.append(sum([i in x for i in y])/len(y))
    elif len(x) > len(y) and len(x) != 0 and x != 'NaN':
        print([i in y for i in x])
        percent_shared.append(sum([i in y for i in x])/len(x))
    elif x == 'NaN' or y == 'NaN':
        continue
    else:
        percent_shared.append(0)

print('Average percent shared is: ' + str(100*np.average(percent_shared)) + '%')

#%% Check if subnetworks are differentially identified - can only compare between contexts
percent_shared = []
for idx in range(0, 5):
    context_shared_subsets = 0
    y = con_A_subnetworks_D1[idx]
    x = con_B_subnetworks_D1[idx]
    if len(y) >= len(x) and len(y) != 0:
        for i in y:
            for j in x:
                if set(i).issubset(set(j)) or set(j).issubset(set(i)):
                    context_shared_subsets += 1
        percent_shared.append(context_shared_subsets/len(y))
    elif len(x) > len(y) and len(x) != 0:
        for i in x:
            for j in y:
                if set(i).issubset(set(j)) or set(j).issubset(set(i)):
                    context_shared_subsets += 1
        percent_shared.append(context_shared_subsets/len(x))
    else: 
        percent_shared.append(0)
    
print('Average percent shared is: ' + str(100*np.average(percent_shared)) + '%')

#%% Check if edges are differentially identified - can only compare between contexts
percent_shared = []
for idx in range(0, 5):
    y = edges_A_D9[idx]
    x = edges_B_D9[idx]
    if len(y) >= len(x) and len(y) != 0:
        percent_shared.append(sum([i in x for i in y])/len(y))
    elif len(x) > len(y) and len(x) != 0:
        percent_shared.append(sum([i in y for i in x])/len(x))
    else:
        percent_shared.append(0)
    
    
print('Average percent shared is: ' + str(100*np.average(percent_shared)) + '%')
print('STDEV: ' + str(100*np.std(percent_shared)))