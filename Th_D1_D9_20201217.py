# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:30:42 2020

@author: Veronica Porubsky

Title: Run batch analyses

Context A - anxiogenic
Context B - neutral
"""
# Todo: add statistics comparing D1 and D9 with D5
# Todo: compute statistics on populations of random networks, not specimen-matched single examples of random networks
import logging
from neuronal_network_graph import DGNetworkGraph as nng
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import scipy
import seaborn as sns

sns.set(style="whitegrid")
my_pal = {"D1_A": "salmon", "D1_B": "darkturquoise", "D5_A": "salmon", "D5_B": "darkturquoise", "D9_A":"salmon", "D9_B":"darkturquoise"}
my_pal = {"rand_D1_A": "grey", "erdos_renyi_D1_A": "black", "D1_A": "salmon", "rand_D1_B": "grey","erdos_renyi_D1_B": "black", "D1_B": "darkturquoise", "rand_D5_A": "grey", "D5_A": "salmon", "rand_D5_B": "grey","D5_B": "darkturquoise", "rand_D9_A": "grey", "D9_A":"salmon", "rand_D9_B": "grey", "D9_B":"darkturquoise"}
my_pal2 = {"rand_D1_A": "lightcoral", "erdos_renyi_D1_A": "lightcoral", "WT_D1_A": "firebrick", "rand_D1_B": "paleturquoise","erdos_renyi_D1_B": "paleturquoise", "WT_D1_B": "cadetblue", "rand_D5_A": "lightcoral", "erdos_renyi_D5_A": "lightcoral",  "WT_D5_A": "firebrick", "rand_D5_B": "paleturquoise","erdos_renyi_D5_B": "paleturquoise","WT_D5_B": "cadetblue", "rand_D9_A": "lightcoral", "erdos_renyi_D9_A": "lightcoral",  "WT_D9_A":"firebrick", "rand_D9_B": "paleturquoise","erdos_renyi_D9_B": "paleturquoise", "WT_D9_B":"cadetblue"}
_log = logging.getLogger(__name__)

#%% Load treated data files - Th
D0_Th = []
D1_Th = ['2-1_D1_smoothed_calcium_traces.csv', '2-2_D1_smoothed_calcium_traces.csv','2-3_D1_smoothed_calcium_traces.csv', '348-1_D1_smoothed_calcium_traces.csv', '349-2_D1_smoothed_calcium_traces.csv', '386-2_D1_smoothed_calcium_traces.csv', '387-4_D1_smoothed_calcium_traces.csv', '396-1_D1_smoothed_calcium_traces.csv', '396-3_D1_smoothed_calcium_traces.csv']
D5_Th =
D9_Th = ['2-1_D9_smoothed_calcium_traces.csv', '2-2_D9_smoothed_calcium_traces.csv','2-3_D9_smoothed_calcium_traces.csv', '348-1_D9_smoothed_calcium_traces.csv', '349-2_D9_smoothed_calcium_traces.csv', '386-2_D9_smoothed_calcium_traces.csv', '387-4_D9_smoothed_calcium_traces.csv', '396-1_D9_smoothed_calcium_traces.csv', '396-3_D9_smoothed_calcium_traces.csv']

all_treated_files = [day1_treated, day9_treated]

#%% All measurements, separating contexts 
threshold = 0.3
random_graph_type = 'rewire_edges' # choose from 'rewire_edges' and 'erdos_renyi'
names = []
data_mat = []

con_A_subnetworks_D1 = []
con_B_subnetworks_D1 = []
con_A_subnetworks_D5 = []
con_B_subnetworks_D5 = []
con_A_subnetworks_D9 = []
con_B_subnetworks_D9 = []

con_A_num_subnetworks_D1 = []
con_B_num_subnetworks_D1 = []
con_A_num_subnetworks_D5 = []
con_B_num_subnetworks_D5 = []
con_A_num_subnetworks_D9 = []
con_B_num_subnetworks_D9 = []

con_A_size_subnetworks_D1 = []
con_B_size_subnetworks_D1 = []
con_A_size_subnetworks_D5 = []
con_B_size_subnetworks_D5 = []
con_A_size_subnetworks_D9 = []
con_B_size_subnetworks_D9 = []

con_A_cc_D1 = []
con_B_cc_D1 = []
con_A_cc_D9 = []
con_B_cc_D9 = []

con_A_hubs_D1 = []
con_B_hubs_D1 = []
con_A_hubs_D5 = []
con_B_hubs_D5 = []
con_A_hubs_D9 = []
con_B_hubs_D9 = []

con_A_num_hubs_D1 = []
con_B_num_hubs_D1 = []
con_A_num_hubs_D5 = []
con_B_num_hubs_D5 = []
con_A_num_hubs_D9 = []
con_B_num_hubs_D9 = []

edges_A_D1 = []
edges_B_D1 = []
edges_A_D5 = []
edges_B_D5 = []
edges_A_D9 = []
edges_B_D9 = []

mouse_id_indices = []

#%% Context A and B
# Loop through all subjects and perform experimental and randomized network analyses
for treatment_group_index in [0,1]:
    for mouse_id_index in range(len(all_treated_files[treatment_group_index])):
        filename = all_treated_files[treatment_group_index][mouse_id_index]
        mouse_id = filename.strip('_smoothed_calcium_traces.csv')

        if treatment_group_index == 0:
            mouse_id_indices.append(mouse_id.replace('_D1', ''))
        
        nn = nng(filename)
        print(f"Executing analyses for {mouse_id}")
        num_neurons = nn.num_neurons

        # Context A and B graphs
        conA = nn.get_context_A_graph(threshold = threshold)
        conB = nn.get_context_B_graph(threshold = threshold)

        # subnetwork analysis
        connected_subnetworks_A = nn.get_context_A_subnetworks(threshold = threshold)
        connected_subnetworks_B = nn.get_context_B_subnetworks(threshold = threshold)

        num_connected_subnetworks_A = len(connected_subnetworks_A)
        len_connected_subnetworks_A = []
        [len_connected_subnetworks_A.append(len(x)) for x in connected_subnetworks_A]

        num_connected_subnetworks_B = len(connected_subnetworks_B)
        len_connected_subnetworks_B = []
        [len_connected_subnetworks_B.append(len(x)) for x in connected_subnetworks_B]

        # clustering coefficient
        cc_A = nn.get_context_A_clustering_coefficient()
        cc_B = nn.get_context_B_clustering_coefficient()

        # hub analysis
        hubs_A, tmp = nn.get_context_A_hubs(threshold=threshold)
        hubs_B, tmp = nn.get_context_B_hubs(threshold=threshold)

        len_hubs_A = len(hubs_A)
        len_hubs_B = len(hubs_B)

        if treatment_group_index == 0:
            # Experimental
            con_A_subnetworks_D1.append(connected_subnetworks_A)
            con_B_subnetworks_D1.append(connected_subnetworks_B)
            con_A_num_subnetworks_D1.append(num_connected_subnetworks_A/num_neurons)
            con_B_num_subnetworks_D1.append(num_connected_subnetworks_B/num_neurons)
            con_A_size_subnetworks_D1.append(np.median(len_connected_subnetworks_A)/num_neurons)
            con_B_size_subnetworks_D1.append(np.median(len_connected_subnetworks_B)/num_neurons)
            edges_A_D1.append(list(conA.edges()))
            edges_B_D1.append(list(conB.edges()))
            con_A_hubs_D1.append(hubs_A)
            con_B_hubs_D1.append(hubs_B)
            con_A_num_hubs_D1.append(len_hubs_A/num_neurons)
            con_B_num_hubs_D1.append(len_hubs_B/num_neurons)
            con_A_cc_D1.append(np.average(cc_A))
            con_B_cc_D1.append(np.average(cc_B))

        elif treatment_group_index == 1:
            con_A_subnetworks_D9.append(connected_subnetworks_A)
            con_B_subnetworks_D9.append(connected_subnetworks_B)
            con_A_num_subnetworks_D9.append(num_connected_subnetworks_A/num_neurons)
            con_B_num_subnetworks_D9.append(num_connected_subnetworks_B/num_neurons)
            con_A_size_subnetworks_D9.append(np.median(len_connected_subnetworks_A)/num_neurons)
            con_B_size_subnetworks_D9.append(np.median(len_connected_subnetworks_B)/num_neurons)
            edges_A_D9.append(list(conA.edges()))
            edges_B_D9.append(list(conB.edges()))
            con_A_hubs_D9.append(hubs_A)
            con_B_hubs_D9.append(hubs_B)
            con_A_num_hubs_D9.append(len_hubs_A/num_neurons)
            con_B_num_hubs_D9.append(len_hubs_B/num_neurons)
            con_A_cc_D9.append(np.average(cc_A))
            con_B_cc_D9.append(np.average(cc_B))

#%% plot clustering coefficient
baseline_inc = {"D1_A": "salmon", "D1_B": "darkturquoise", "D9_A":"salmon", "D9_B":"darkturquoise"}

labels = ['D1_A', 'D1_B', 'D9_A', 'D9_B']
raw = [con_A_cc_D1, con_B_cc_D1, con_A_cc_D9, con_B_cc_D9]
data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels, index=mouse_id_indices)

sns.swarmplot(data=data, color = 'k');
sns.boxplot(data=data, whis = 1.5, palette=baseline_inc);

plt.ylim((0,0.03))
plt.ylabel('Normalized clustering coefficient')
plt.title('Edge weight threshold: ' + str(threshold))
plt.show()

#%% Plot baseline vs Day 1, 5, 9 Hubs
baseline_inc = {"D1_A": "salmon", "D1_B": "darkturquoise", "D9_A":"salmon", "D9_B":"darkturquoise"}

labels = ['D1_A', 'D1_B', 'D9_A', 'D9_B']
raw = [con_A_num_hubs_D1, con_B_num_hubs_D1, con_A_num_hubs_D9, con_B_num_hubs_D9]
data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels, index=mouse_id_indices)

sns.swarmplot(data=data, color = 'k');
sns.boxplot(data=data, whis = 1.5, palette=baseline_inc);

plt.ylabel('Normalized hub count')
plt.title('Edge weight threshold: ' + str(threshold))
plt.show()

#%%  Plot baseline vs Day 1, 9 number of subnetworks
labels = ['D1_A', 'D1_B', 'D9_A', 'D9_B']
raw = [con_A_num_subnetworks_D1, con_B_num_subnetworks_D1, con_A_num_subnetworks_D9, con_B_num_subnetworks_D9]
data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels, index=mouse_id_indices)

sns.swarmplot(data=data, color = 'k');
sns.boxplot(data=data, whis = 1.5, palette=baseline_inc);

plt.ylabel('Normalized subnetwork count')
plt.title('Edge weight threshold: ' + str(threshold))
plt.show()

#%%  Plot baseline vs Day 1, 5, 9 size of subnetworks
labels = ['D1_A', 'D1_B', 'D9_A', 'D9_B']
raw = [con_A_size_subnetworks_D1, con_B_size_subnetworks_D1, con_A_size_subnetworks_D9, con_B_size_subnetworks_D9]
data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels, index=mouse_id_indices)

sns.swarmplot(data=data, color = 'k');
sns.boxplot(data=data, whis = 1.5, palette=baseline_inc);

plt.ylabel('Normalized subnetwork size')
plt.title('Edge weight threshold: ' + str(threshold))
plt.show()

#%% plot hub metrics with seaborn
labels = ['D1_A', 'D1_B','D9_A', 'D9_B']
raw = [con_A_num_hubs_D1, con_B_num_hubs_D1, con_A_num_hubs_D9, con_B_num_hubs_D9]

#plt.figure(figsize=(20,10))
data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels, index=mouse_id_indices)
sns.swarmplot(data=data, color = 'k');
sns.boxplot(data=data, whis = 1.5, palette=my_pal);

plt.ylabel('Normalized hub count')
plt.title('Edge weight threshold: ' + str(threshold))
plt.show()

#%% Compute hub statistics 
D1_A_B = scipy.stats.ttest_rel(con_A_num_hubs_D1, con_B_num_hubs_D1)
D9_A_B = scipy.stats.ttest_rel(con_A_num_hubs_D9, con_B_num_hubs_D9)
A_D1_D9 = scipy.stats.ttest_rel(con_A_num_hubs_D1, con_A_num_hubs_D9)
B_D1_D9 = scipy.stats.ttest_rel(con_B_num_hubs_D1, con_B_num_hubs_D9)

test_stats = ['D1 A and B: ' + str(D1_A_B.pvalue),'D9 A and B: ' + str(D9_A_B.pvalue),'A D1 and D9: ' + str(A_D1_D9.pvalue),'B D1 and D9: ' + str(B_D1_D9.pvalue)]

_log.info(f"Hub statistical test results: {test_stats}")
print('')
print('Hub statistical test results:')
[print(x) for x in test_stats] 
print('')


#%% plot subnetwork number metrics with seaborn 
labels = ['D1_A', 'D1_B', 'D9_A', 'D9_B']
raw = [con_A_num_subnetworks_D1, con_B_num_subnetworks_D1, con_A_num_subnetworks_D9, con_B_num_subnetworks_D9]

data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels, index=mouse_id_indices)

fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
sns.boxplot(data=data, whis = 1.5, palette=my_pal);
sns.swarmplot(data=data, color = 'k');
plt.ylabel('Normalized subnetwork count')
plt.title('Edge weight threshold: ' + str(threshold))
plt.show()

#%% plot subnetwork size  metrics with seaborn 
labels = ['D1_A', 'D1_B', 'D9_A', 'D9_B']
raw = [con_A_size_subnetworks_D1, con_B_size_subnetworks_D1, con_A_size_subnetworks_D9, con_B_size_subnetworks_D9]


data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels, index=mouse_id_indices)

#fig, ax = plt.subplots(figsize=(20,10))
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
sns.boxplot(data=data, whis = 1.5, palette=my_pal);
sns.swarmplot(data=data, color = 'k');
plt.ylabel('Normalized median subnetwork size')
plt.title('Edge weight threshold: ' + str(threshold))
plt.show()

#%% Compute subnetwork number statistics 
D1_A_B = scipy.stats.ttest_rel(con_A_num_subnetworks_D1, con_B_num_subnetworks_D1)
D9_A_B = scipy.stats.ttest_rel(con_A_num_subnetworks_D9, con_B_num_subnetworks_D9)
A_D1_D9 = scipy.stats.ttest_rel(con_A_num_subnetworks_D1, con_A_num_subnetworks_D9)
B_D1_D9 = scipy.stats.ttest_rel(con_B_num_subnetworks_D1, con_B_num_subnetworks_D9)

test_stats = ['D1 A and B: ' + str(D1_A_B.pvalue),'D9 A and B: ' + str(D9_A_B.pvalue),'A D1 and D9: ' + str(A_D1_D9.pvalue),'B D1 and D9: ' + str(B_D1_D9.pvalue)]
print('Subnetwork count statistical test results:')
[print(x) for x in test_stats] 
print('')

#%% Compute subnetwork size statistics 
D1_A_B = scipy.stats.ttest_rel(con_A_size_subnetworks_D1, con_B_size_subnetworks_D1)
D9_A_B = scipy.stats.ttest_rel(con_A_size_subnetworks_D9, con_B_size_subnetworks_D9)
A_D1_D9 = scipy.stats.ttest_rel(con_A_size_subnetworks_D1, con_A_size_subnetworks_D9)
B_D1_D9 = scipy.stats.ttest_rel(con_B_size_subnetworks_D1, con_B_size_subnetworks_D9)

test_stats = ['D1 A and B: ' + str(D1_A_B.pvalue),'D9 A and B: ' + str(D9_A_B.pvalue),'A D1 and D9: ' + str(A_D1_D9.pvalue),'B D1 and D9: ' + str(B_D1_D9.pvalue)]
print('Subnetwork size statistical test results:')
[print(x) for x in test_stats] 
print('')

#%% Check if hubs are differentially identified - can only compare between contexts
percent_shared = []
for idx in range(0,5):
    y = con_A_hubs_D1[idx]
    x = con_B_hubs_D1[idx]
    if len(y) >= len(x) and len(y) != 0 and y != 'NaN':
        percent_shared.append(sum([i in x for i in y])/len(y))
    elif len(x) > len(y) and len(x) != 0 and x != 'NaN':
        percent_shared.append(sum([i in y for i in x])/len(x))
    elif x == 'NaN' or y == 'NaN':
        continue
    else:
        percent_shared.append(0)

print('Average percentage of hubs shared between Context A and B is: ' + str(100*np.average(percent_shared)) + '%')

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
    
print('Average percentage of subnetworks shared between Context A and B is: ' + str(100*np.average(percent_shared)) + '%')

#%% Check if edges are differentially identified - can only compare between contexts
percent_shared = []
for idx in range(0, 5):
    y = edges_A_D1[idx]
    x = edges_B_D1[idx]
    if len(y) >= len(x) and len(y) != 0:
        percent_shared.append(sum([i in x for i in y])/len(y))
    elif len(x) > len(y) and len(x) != 0:
        percent_shared.append(sum([i in y for i in x])/len(x))
    else:
        percent_shared.append(0)
    
    
print('Average percentage of edges shared between Context A and B is: ' + str(100*np.average(percent_shared)) + '%')
#print('STDEV: ' + str(100*np.std(percent_shared)))
print('')

#%% Subnetwork count: Check for statistical significance between measured networks and randomized networks
D1_A = scipy.stats.ttest_rel(con_A_num_subnetworks_D1, rand_con_A_num_subnetworks_D1)
D1_B = scipy.stats.ttest_rel(con_B_num_subnetworks_D1, rand_con_B_num_subnetworks_D1)
D5_A = scipy.stats.ttest_rel(con_A_num_subnetworks_D5, rand_con_A_num_subnetworks_D5)
D5_B = scipy.stats.ttest_rel(con_B_num_subnetworks_D5, rand_con_B_num_subnetworks_D5)
D9_A = scipy.stats.ttest_rel(con_A_num_subnetworks_D9, rand_con_A_num_subnetworks_D9)
D9_B = scipy.stats.ttest_rel(con_B_num_subnetworks_D9, rand_con_B_num_subnetworks_D9)

test_stats = ['D1 A and D1 randomized A: ' + str(D1_A.pvalue), 'D1 B and D1 randomized B: ' + str(D1_B.pvalue),\
              'D5 A and D5 randomized A: ' + str(D5_A.pvalue), 'D5 B and D5 randomized B: ' + str(D5_B.pvalue),\
              'D9 A and D9 randomized A: ' + str(D9_A.pvalue), 'D9 B and D9 randomized B: ' + str(D9_B.pvalue),]
print('Subnetwork count statistical test results - compared to randomized network:')
[print(x) for x in test_stats]
print('')

#%% Subnetwork size: Check for statistical significance between measured networks and randomized networks
D1_A = scipy.stats.ttest_rel(con_A_size_subnetworks_D1, rand_con_A_size_subnetworks_D1)
D1_B = scipy.stats.ttest_rel(con_B_size_subnetworks_D1, rand_con_B_size_subnetworks_D1)
D5_A = scipy.stats.ttest_rel(con_A_size_subnetworks_D5, rand_con_A_size_subnetworks_D5)
D5_B = scipy.stats.ttest_rel(con_B_size_subnetworks_D5, rand_con_B_size_subnetworks_D5)
D9_A = scipy.stats.ttest_rel(con_A_size_subnetworks_D9, rand_con_A_size_subnetworks_D9)
D9_B = scipy.stats.ttest_rel(con_B_size_subnetworks_D9, rand_con_B_size_subnetworks_D9)

test_stats = ['D1 A and D1 randomized A: ' + str(D1_A.pvalue), 'D1 B and D1 randomized B: ' + str(D1_B.pvalue),\
              'D5 A and D5 randomized A: ' + str(D5_A.pvalue), 'D5 B and D5 randomized B: ' + str(D5_B.pvalue),\
              'D9 A and D9 randomized A: ' + str(D9_A.pvalue), 'D9 B and D9 randomized B: ' + str(D9_B.pvalue),]
print('Subnetwork size statistical test results - compared to randomized network:')
[print(x) for x in test_stats]
print('')

#%% Hubs: Check for statistical significance between measured networks and randomized networks
D1_A = scipy.stats.ttest_rel(con_A_num_hubs_D1, rand_con_A_num_hubs_D1)
D1_B = scipy.stats.ttest_rel(con_B_num_hubs_D1, rand_con_B_num_hubs_D1)
D5_A = scipy.stats.ttest_rel(con_A_num_hubs_D5, rand_con_A_num_hubs_D5)
D5_B = scipy.stats.ttest_rel(con_B_num_hubs_D5, rand_con_B_num_hubs_D5)
D9_A = scipy.stats.ttest_rel(con_A_num_hubs_D9, rand_con_A_num_hubs_D9)
D9_B = scipy.stats.ttest_rel(con_B_num_hubs_D9, rand_con_B_num_hubs_D9)

test_stats = ['D1 A and D1 randomized A: ' + str(D1_A.pvalue), 'D1 B and D1 randomized B: ' + str(D1_B.pvalue),\
              'D5 A and D5 randomized A: ' + str(D5_A.pvalue), 'D5 B and D5 randomized B: ' + str(D5_B.pvalue),\
              'D9 A and D9 randomized A: ' + str(D9_A.pvalue), 'D9 B and D9 randomized B: ' + str(D9_B.pvalue),]
print('Hub count statistical test results - compared to randomized network:')
[print(x) for x in test_stats]
print('')

