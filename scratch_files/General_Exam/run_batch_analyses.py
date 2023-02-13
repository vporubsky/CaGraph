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
from ca_graph import neuronal_network_graph as nng
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import scipy
import seaborn as sns
import networkx as nx
sns.set(style="whitegrid")
my_pal = {"D1_A": "salmon", "D1_B": "darkturquoise", "D5_A": "salmon", "D5_B": "darkturquoise", "D9_A":"salmon", "D9_B":"darkturquoise"}
my_pal = {"rand_D1_A": "grey", "erdos_renyi_D1_A": "black", "D1_A": "salmon", "rand_D1_B": "grey","erdos_renyi_D1_B": "black", "D1_B": "darkturquoise", "rand_D5_A": "grey", "D5_A": "salmon", "rand_D5_B": "grey","D5_B": "darkturquoise", "rand_D9_A": "grey", "D9_A":"salmon", "rand_D9_B": "grey", "D9_B":"darkturquoise"}
my_pal2 = {"rand_D1_A": "lightcoral", "erdos_renyi_D1_A": "lightcoral", "WT_D1_A": "firebrick", "rand_D1_B": "paleturquoise","erdos_renyi_D1_B": "paleturquoise", "WT_D1_B": "cadetblue", "rand_D5_A": "lightcoral", "erdos_renyi_D5_A": "lightcoral",  "WT_D5_A": "firebrick", "rand_D5_B": "paleturquoise","erdos_renyi_D5_B": "paleturquoise","WT_D5_B": "cadetblue", "rand_D9_A": "lightcoral", "erdos_renyi_D9_A": "lightcoral",  "WT_D9_A":"firebrick", "rand_D9_B": "paleturquoise","erdos_renyi_D9_B": "paleturquoise", "WT_D9_B":"cadetblue"}
_log = logging.getLogger(__name__)

#%% Load untreated data files - WT
day1_untreated = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv', '1055-3_D1_smoothed_calcium_traces.csv', '1055-4_D1_smoothed_calcium_traces.csv', '14-0_D1_smoothed_calcium_traces.csv', '122-1_D1_smoothed_calcium_traces.csv', '122-2_D1_smoothed_calcium_traces.csv', '122-3_D1_smoothed_calcium_traces.csv']#, '124-2_D1_smoothed_calcium_traces.csv']
day5_untreated = ['1055-1_D5_smoothed_calcium_traces.csv', '1055-2_D5_smoothed_calcium_traces.csv','1055-3_D5_smoothed_calcium_traces.csv', '1055-4_D5_smoothed_calcium_traces.csv', '14-0_D5_smoothed_calcium_traces.csv']
day9_untreated = ['1055-1_D9_smoothed_calcium_traces.csv', '1055-2_D9_smoothed_calcium_traces.csv','1055-3_D9_smoothed_calcium_traces.csv', '1055-4_D9_smoothed_calcium_traces.csv', '14-0_D9_smoothed_calcium_traces.csv', '122-1_D9_smoothed_calcium_traces.csv', '122-2_D9_smoothed_calcium_traces.csv', '122-3_D9_smoothed_calcium_traces.csv']#, '124-2_D9_smoothed_calcium_traces.csv']
day0_untreated = ['1055-1_D0_smoothed_calcium_traces.csv','1055-2_D0_smoothed_calcium_traces.csv','1055-3_D0_smoothed_calcium_traces.csv','1055-4_D0_smoothed_calcium_traces.csv','14-0_D0_smoothed_calcium_traces.csv']
all_untreated_files = [day1_untreated, day5_untreated, day9_untreated]

#%% Load treated data files - Th
day1_treated = ['2-1_D1_smoothed_calcium_traces.csv', '2-2_D1_smoothed_calcium_traces.csv','2-3_D1_smoothed_calcium_traces.csv', '348-1_D1_smoothed_calcium_traces.csv', '349-2_D1_smoothed_calcium_traces.csv', '386-2_D1_smoothed_calcium_traces.csv', '387-4_D1_smoothed_calcium_traces.csv', '396-1_D1_smoothed_calcium_traces.csv']
day5_treated = ['2-1_D5_smoothed_calcium_traces.csv', '2-2_D5_smoothed_calcium_traces.csv','2-3_D5_smoothed_calcium_traces.csv', '348-1_D5_smoothed_calcium_traces.csv', '349-2_D5_smoothed_calcium_traces.csv', '386-2_D5_smoothed_calcium_traces.csv', '387-4_D5_smoothed_calcium_traces.csv', '396-1_D5_smoothed_calcium_traces.csv']
day9_treated = ['2-1_D9_smoothed_calcium_traces.csv', '2-2_D9_smoothed_calcium_traces.csv','2-3_D9_smoothed_calcium_traces.csv', '348-1_D9_smoothed_calcium_traces.csv', '349-2_D9_smoothed_calcium_traces.csv', '386-2_D9_smoothed_calcium_traces.csv', '387-4_D9_smoothed_calcium_traces.csv', '396-1_D9_smoothed_calcium_traces.csv']

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

rand_con_A_subnetworks_D1 = []
rand_con_B_subnetworks_D1 = []
rand_con_A_subnetworks_D5 = []
rand_con_B_subnetworks_D5 = []
rand_con_A_subnetworks_D9 = []
rand_con_B_subnetworks_D9 = []

rand_con_A_num_subnetworks_D1 = []
rand_con_B_num_subnetworks_D1 = []
rand_con_A_num_subnetworks_D5 = []
rand_con_B_num_subnetworks_D5 = []
rand_con_A_num_subnetworks_D9 = []
rand_con_B_num_subnetworks_D9 = []

rand_con_A_size_subnetworks_D1 = []
rand_con_B_size_subnetworks_D1 = []
rand_con_A_size_subnetworks_D5 = []
rand_con_B_size_subnetworks_D5 = []
rand_con_A_size_subnetworks_D9 = []
rand_con_B_size_subnetworks_D9 = []

erdos_renyi_con_A_subnetworks_D1 = []
erdos_renyi_con_B_subnetworks_D1 = []
erdos_renyi_con_A_subnetworks_D5 = []
erdos_renyi_con_B_subnetworks_D5 = []
erdos_renyi_con_A_subnetworks_D9 = []
erdos_renyi_con_B_subnetworks_D9 = []

erdos_renyi_con_A_num_subnetworks_D1 = []
erdos_renyi_con_B_num_subnetworks_D1 = []
erdos_renyi_con_A_num_subnetworks_D5 = []
erdos_renyi_con_B_num_subnetworks_D5 = []
erdos_renyi_con_A_num_subnetworks_D9 = []
erdos_renyi_con_B_num_subnetworks_D9 = []

erdos_renyi_con_A_size_subnetworks_D1 = []
erdos_renyi_con_B_size_subnetworks_D1 = []
erdos_renyi_con_A_size_subnetworks_D5 = []
erdos_renyi_con_B_size_subnetworks_D5 = []
erdos_renyi_con_A_size_subnetworks_D9 = []
erdos_renyi_con_B_size_subnetworks_D9 = []

rand_con_A_hubs_D1 = []
rand_con_B_hubs_D1 = []
rand_con_A_hubs_D5 = []
rand_con_B_hubs_D5 = []
rand_con_A_hubs_D9 = []
rand_con_B_hubs_D9 = []

rand_con_A_num_hubs_D1 = []
rand_con_B_num_hubs_D1 = []
rand_con_A_num_hubs_D5 = []
rand_con_B_num_hubs_D5 = []
rand_con_A_num_hubs_D9 = []
rand_con_B_num_hubs_D9 = []

rand_edges_A_D1 = []
rand_edges_B_D1 = []
rand_edges_A_D5 = []
rand_edges_B_D5 = []
rand_edges_A_D9 = []
rand_edges_B_D9 = []

mouse_id_indices = []

#%% Context A and B
# Loop through all subjects and perform experimental and randomized network analyses
for treatment_group_index in [0,1,2]:
    for mouse_id_index in range(len(all_untreated_files[treatment_group_index])):
        filename = all_untreated_files[treatment_group_index][mouse_id_index]
        mouse_id = filename.strip('_smoothed_calcium_traces.csv')

        if treatment_group_index == 0:
            mouse_id_indices.append(mouse_id.replace('_D1', ''))
        
        nn = nng(filename)
        print(f"Executing analyses for {mouse_id}")
        num_neurons = nn.num_neurons

        # Context A and B graphs
        conA = nn.get_context_A_graph(threshold = threshold)
        conB = nn.get_context_B_graph(threshold = threshold)

        # Context A and B randomized graphs -- randomize edges of existing graph
        if random_graph_type == 'rewire_edges':
            rand_conA = nn.get_random_context_A_graph(threshold=threshold)
            rand_conB = nn.get_random_context_B_graph(threshold=threshold)
            # Todo: remove in final version,
            erdos_renyi_conA = nn.get_erdos_renyi_graph(graph=conA, threshold=threshold)
            erdos_renyi_conB = nn.get_erdos_renyi_graph(graph=conB, threshold=threshold)

        # Context A and B randomized graphs -- erdos-renyi
        elif random_graph_type == 'erdos_renyi':
            rand_conA = nn.get_erdos_renyi_graph(graph=conA, threshold=threshold)
            rand_conB = nn.get_erdos_renyi_graph(graph=conB, threshold=threshold)

        # subnetwork analysis
        connected_subnetworks_A = nn.get_context_A_subnetworks(threshold = threshold)
        connected_subnetworks_B = nn.get_context_B_subnetworks(threshold = threshold)

        num_connected_subnetworks_A = len(connected_subnetworks_A)
        len_connected_subnetworks_A = []
        [len_connected_subnetworks_A.append(len(x)) for x in connected_subnetworks_A]

        num_connected_subnetworks_B = len(connected_subnetworks_B)
        len_connected_subnetworks_B = []
        [len_connected_subnetworks_B.append(len(x)) for x in connected_subnetworks_B]

        # random subnetwork analysis
        rand_connected_subnetworks_A = nn.get_subnetworks(graph=rand_conA, threshold=threshold)
        rand_connected_subnetworks_B = nn.get_subnetworks(graph=rand_conB, threshold=threshold)

        rand_num_connected_subnetworks_A = len(rand_connected_subnetworks_A)
        rand_len_connected_subnetworks_A = []
        [rand_len_connected_subnetworks_A.append(len(x)) for x in rand_connected_subnetworks_A]

        rand_num_connected_subnetworks_B = len(rand_connected_subnetworks_B)
        rand_len_connected_subnetworks_B = []
        [rand_len_connected_subnetworks_B.append(len(x)) for x in rand_connected_subnetworks_B]

        # Todo: remove in final version
        # erdos-renyi
        erdos_renyi_connected_subnetworks_A = nn.get_subnetworks(graph=erdos_renyi_conA, threshold=threshold)
        erdos_renyi_connected_subnetworks_B = nn.get_subnetworks(graph=erdos_renyi_conB, threshold=threshold)

        erdos_renyi_num_connected_subnetworks_A = len(erdos_renyi_connected_subnetworks_A)
        erdos_renyi_len_connected_subnetworks_A = []
        [erdos_renyi_len_connected_subnetworks_A.append(len(x)) for x in erdos_renyi_connected_subnetworks_A]

        erdos_renyi_num_connected_subnetworks_B = len(erdos_renyi_connected_subnetworks_B)
        erdos_renyi_len_connected_subnetworks_B = []
        [erdos_renyi_len_connected_subnetworks_B.append(len(x)) for x in erdos_renyi_connected_subnetworks_B]

        # hub analysis
        hubs_A, tmp = nn.get_context_A_hubs(threshold=threshold)
        hubs_B, tmp = nn.get_context_B_hubs(threshold=threshold)

        len_hubs_A = len(hubs_A)
        len_hubs_B = len(hubs_B)

        # random hub analysis
        rand_hubs_A, tmp = nn.get_hubs(threshold=threshold, graph=rand_conA)
        rand_hubs_B, tmp = nn.get_hubs(threshold=threshold, graph=rand_conB)

        rand_len_hubs_A = len(rand_hubs_A)
        rand_len_hubs_B = len(rand_hubs_B)

        if treatment_group_index == 0:
            # Experimental
            con_A_subnetworks_D1.append(connected_subnetworks_A)
            con_B_subnetworks_D1.append(connected_subnetworks_B)
            con_A_num_subnetworks_D1.append(num_connected_subnetworks_A/num_neurons)
            con_B_num_subnetworks_D1.append(num_connected_subnetworks_B/num_neurons)
            con_A_size_subnetworks_D1.append(np.average(len_connected_subnetworks_A)/num_neurons)
            con_B_size_subnetworks_D1.append(np.average(len_connected_subnetworks_B)/num_neurons)
            edges_A_D1.append(list(conA.edges()))
            edges_B_D1.append(list(conB.edges()))
            con_A_hubs_D1.append(hubs_A)
            con_B_hubs_D1.append(hubs_B)
            con_A_num_hubs_D1.append(len_hubs_A/num_neurons)
            con_B_num_hubs_D1.append(len_hubs_B/num_neurons)
            # Randomized
            rand_con_A_subnetworks_D1.append(rand_connected_subnetworks_A)
            rand_con_B_subnetworks_D1.append(rand_connected_subnetworks_B)
            rand_con_A_num_subnetworks_D1.append(rand_num_connected_subnetworks_A/num_neurons)
            rand_con_B_num_subnetworks_D1.append(rand_num_connected_subnetworks_B/num_neurons)
            rand_con_A_size_subnetworks_D1.append(np.average(rand_len_connected_subnetworks_A)/num_neurons)
            rand_con_B_size_subnetworks_D1.append(np.average(rand_len_connected_subnetworks_B)/num_neurons)
            rand_edges_A_D1.append(list(rand_conA.edges()))
            rand_edges_B_D1.append(list(rand_conB.edges()))
            rand_con_A_hubs_D1.append(rand_hubs_A)
            rand_con_B_hubs_D1.append(rand_hubs_B)
            rand_con_A_num_hubs_D1.append(rand_len_hubs_A/num_neurons)
            rand_con_B_num_hubs_D1.append(rand_len_hubs_B/num_neurons)
            # Erdos-Renyi
            erdos_renyi_con_A_subnetworks_D1.append(erdos_renyi_connected_subnetworks_A)
            erdos_renyi_con_B_subnetworks_D1.append(erdos_renyi_connected_subnetworks_B)
            erdos_renyi_con_A_num_subnetworks_D1.append(erdos_renyi_num_connected_subnetworks_A/num_neurons)
            erdos_renyi_con_B_num_subnetworks_D1.append(erdos_renyi_num_connected_subnetworks_B/num_neurons)
            erdos_renyi_con_A_size_subnetworks_D1.append(np.average(erdos_renyi_len_connected_subnetworks_A)/num_neurons)
            erdos_renyi_con_B_size_subnetworks_D1.append(np.average(erdos_renyi_len_connected_subnetworks_B)/num_neurons)

        elif treatment_group_index == 1:
            con_A_subnetworks_D5.append(connected_subnetworks_A)
            con_B_subnetworks_D5.append(connected_subnetworks_B)
            con_A_num_subnetworks_D5.append(num_connected_subnetworks_A/num_neurons)
            con_B_num_subnetworks_D5.append(num_connected_subnetworks_B/num_neurons)
            con_A_size_subnetworks_D5.append(np.average(len_connected_subnetworks_A)/num_neurons)
            con_B_size_subnetworks_D5.append(np.average(len_connected_subnetworks_B)/num_neurons)
            edges_A_D5.append(list(conA.edges()))
            edges_B_D5.append(list(conB.edges()))
            con_A_hubs_D5.append(hubs_A)
            con_B_hubs_D5.append(hubs_B)
            con_A_num_hubs_D5.append(len_hubs_A/num_neurons)
            con_B_num_hubs_D5.append(len_hubs_B/num_neurons)
            # Randomized
            rand_con_A_subnetworks_D5.append(rand_connected_subnetworks_A)
            rand_con_B_subnetworks_D5.append(rand_connected_subnetworks_B)
            rand_con_A_num_subnetworks_D5.append(rand_num_connected_subnetworks_A/num_neurons)
            rand_con_B_num_subnetworks_D5.append(rand_num_connected_subnetworks_B/num_neurons)
            rand_con_A_size_subnetworks_D5.append(np.average(rand_len_connected_subnetworks_A)/num_neurons)
            rand_con_B_size_subnetworks_D5.append(np.average(rand_len_connected_subnetworks_B)/num_neurons)
            rand_edges_A_D5.append(list(rand_conA.edges()))
            rand_edges_B_D5.append(list(rand_conB.edges()))
            rand_con_A_hubs_D5.append(rand_hubs_A)
            rand_con_B_hubs_D5.append(rand_hubs_B)
            rand_con_A_num_hubs_D5.append(rand_len_hubs_A/num_neurons)
            rand_con_B_num_hubs_D5.append(rand_len_hubs_B/num_neurons)
            # Erdos-Renyi
            erdos_renyi_con_A_subnetworks_D5.append(erdos_renyi_connected_subnetworks_A)
            erdos_renyi_con_B_subnetworks_D5.append(erdos_renyi_connected_subnetworks_B)
            erdos_renyi_con_A_num_subnetworks_D5.append(erdos_renyi_num_connected_subnetworks_A/num_neurons)
            erdos_renyi_con_B_num_subnetworks_D5.append(erdos_renyi_num_connected_subnetworks_B/num_neurons)
            erdos_renyi_con_A_size_subnetworks_D5.append(np.average(erdos_renyi_len_connected_subnetworks_A)/num_neurons)
            erdos_renyi_con_B_size_subnetworks_D5.append(np.average(erdos_renyi_len_connected_subnetworks_B)/num_neurons)

        elif treatment_group_index == 2:
            con_A_subnetworks_D9.append(connected_subnetworks_A)
            con_B_subnetworks_D9.append(connected_subnetworks_B)
            con_A_num_subnetworks_D9.append(num_connected_subnetworks_A/num_neurons)
            con_B_num_subnetworks_D9.append(num_connected_subnetworks_B/num_neurons)
            con_A_size_subnetworks_D9.append(np.average(len_connected_subnetworks_A)/num_neurons)
            con_B_size_subnetworks_D9.append(np.average(len_connected_subnetworks_B)/num_neurons) 
            edges_A_D9.append(list(conA.edges()))
            edges_B_D9.append(list(conB.edges()))
            con_A_hubs_D9.append(hubs_A)
            con_B_hubs_D9.append(hubs_B)
            con_A_num_hubs_D9.append(len_hubs_A/num_neurons)
            con_B_num_hubs_D9.append(len_hubs_B/num_neurons)
            # Randomized
            rand_con_A_subnetworks_D9.append(rand_connected_subnetworks_A)
            rand_con_B_subnetworks_D9.append(rand_connected_subnetworks_B)
            rand_con_A_num_subnetworks_D9.append(rand_num_connected_subnetworks_A/num_neurons)
            rand_con_B_num_subnetworks_D9.append(rand_num_connected_subnetworks_B/num_neurons)
            rand_con_A_size_subnetworks_D9.append(np.average(rand_len_connected_subnetworks_A)/num_neurons)
            rand_con_B_size_subnetworks_D9.append(np.average(rand_len_connected_subnetworks_B)/num_neurons)
            rand_edges_A_D9.append(list(rand_conA.edges()))
            rand_edges_B_D9.append(list(rand_conB.edges()))
            rand_con_A_hubs_D9.append(rand_hubs_A)
            rand_con_B_hubs_D9.append(rand_hubs_B)
            rand_con_A_num_hubs_D9.append(rand_len_hubs_A/num_neurons)
            rand_con_B_num_hubs_D9.append(rand_len_hubs_B/num_neurons)
            # Erdos-Renyi
            erdos_renyi_con_A_subnetworks_D9.append(erdos_renyi_connected_subnetworks_A)
            erdos_renyi_con_B_subnetworks_D9.append(erdos_renyi_connected_subnetworks_B)
            erdos_renyi_con_A_num_subnetworks_D9.append(erdos_renyi_num_connected_subnetworks_A/num_neurons)
            erdos_renyi_con_B_num_subnetworks_D9.append(erdos_renyi_num_connected_subnetworks_B/num_neurons)
            erdos_renyi_con_A_size_subnetworks_D9.append(np.average(erdos_renyi_len_connected_subnetworks_A)/num_neurons)
            erdos_renyi_con_B_size_subnetworks_D9.append(np.average(erdos_renyi_len_connected_subnetworks_B)/num_neurons)
#%% Baseline computation
subnetworks_D0 = []
num_subnetworks_D0 = []
size_subnetworks_D0 = []
hubs_D0 = []
num_hubs_D0 = []
edges_D0 = []

for filename in day0_untreated:
        mouse_id = filename.strip('_smoothed_calcium_traces.csv')

        nn = nng(filename)
        print(f"Executing Day 0 analyses for {mouse_id}")
        num_neurons = nn.num_neurons
        G = nn.get_network_graph(threshold=threshold)

        connected_subnetworks = nn.get_subnetworks(threshold = threshold)
        num_connected_subnetworks = len(connected_subnetworks)
        len_connected_subnetworks = []
        [len_connected_subnetworks.append(len(x)) for x in connected_subnetworks]

        hubs, tmp = nn.get_hubs(threshold=threshold)
        len_hubs = len(hubs)

        subnetworks_D0.append(connected_subnetworks)
        num_subnetworks_D0.append(num_connected_subnetworks / num_neurons)
        size_subnetworks_D0.append(np.average(len_connected_subnetworks) / num_neurons)
        edges_D0.append(list(G.edges()))
        hubs_D0.append(hubs)
        num_hubs_D0.append(len_hubs / num_neurons)

#%% Plot baseline vs Day 1, 5, 9 Hubs
baseline_inc = {"D0": "grey", "D1_A": "salmon", "D1_B": "darkturquoise", "D5_A": "salmon", "D5_B": "darkturquoise", "D9_A":"salmon", "D9_B":"darkturquoise"}

labels = ['D0', 'D1_A', 'D1_B', 'D5_A', 'D5_B', 'D9_A', 'D9_B']
raw = [num_hubs_D0, con_A_num_hubs_D1, con_B_num_hubs_D1, con_A_num_hubs_D5, con_B_num_hubs_D5, con_A_num_hubs_D9, con_B_num_hubs_D9]
data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels, index=mouse_id_indices)

sns.swarmplot(data=data, color = 'k');
sns.boxplot(data=data, whis = 1.5, palette=baseline_inc);

plt.ylabel('Normalized hub count')
plt.title('Edge weight threshold: ' + str(threshold))
plt.show()

#%%  Plot baseline vs Day 1, 5, 9 number of subnetworks
labels = ['D0', 'D1_A', 'D1_B', 'D5_A', 'D5_B', 'D9_A', 'D9_B']
raw = [num_subnetworks_D0, con_A_num_subnetworks_D1, con_B_num_subnetworks_D1, con_A_num_subnetworks_D5, con_B_num_subnetworks_D5, con_A_num_subnetworks_D9, con_B_num_subnetworks_D9]
data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels, index=mouse_id_indices)

sns.swarmplot(data=data, color = 'k');
sns.boxplot(data=data, whis = 1.5, palette=baseline_inc);

plt.ylabel('Normalized subnetwork count')
plt.title('Edge weight threshold: ' + str(threshold))
plt.show()

#%%  Plot baseline vs Day 1, 5, 9 size of subnetworks
labels = ['D0', 'D1_A', 'D1_B', 'D5_A', 'D5_B', 'D9_A', 'D9_B']
raw = [size_subnetworks_D0, con_A_size_subnetworks_D1, con_B_size_subnetworks_D1, con_A_size_subnetworks_D5, con_B_size_subnetworks_D5, con_A_size_subnetworks_D9, con_B_size_subnetworks_D9]
data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels, index=mouse_id_indices)

sns.swarmplot(data=data, color = 'k');
sns.boxplot(data=data, whis = 1.5, palette=baseline_inc);

plt.ylabel('Normalized subnetwork size')
plt.title('Edge weight threshold: ' + str(threshold))
plt.show()

#%% plot randomized vs. wildtype
labels = ['rand_D1_A', 'erdos_renyi_D1_A', 'WT_D1_A', 'rand_D1_B', 'erdos_renyi_D1_B', 'WT_D1_B']
raw = [rand_con_A_num_subnetworks_D1, erdos_renyi_con_A_num_subnetworks_D1, con_A_num_subnetworks_D1, rand_con_B_num_subnetworks_D1, erdos_renyi_con_B_num_subnetworks_D1, con_B_num_subnetworks_D1]

plt.figure(figsize=(12,5))
data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels, index=mouse_id_indices)
sns.swarmplot(data=data, color = 'k');
sns.boxplot(data=data, whis = 1.5, palette=my_pal2);
plt.ylabel('Normalized subnetwork count')
plt.title('Number of subnetworks, edge weight threshold: ' + str(threshold))
plt.show()

#%% plot hub metrics with seaborn
labels = ['D1_A', 'D1_B', 'D5_A', 'D5_B', 'D9_A', 'D9_B']
raw = [con_A_num_hubs_D1, con_B_num_hubs_D1, con_A_num_hubs_D5, con_B_num_hubs_D5, con_A_num_hubs_D9, con_B_num_hubs_D9]

#labels = ['rand_D1_A', 'D1_A', 'rand_D1_B', 'D1_B', 'rand_D5_A','D5_A', 'rand_D5_B','D5_B', 'rand_D9_A','D9_A', 'rand_D9_B', 'D9_B']
#raw = [rand_con_A_num_hubs_D1, con_A_num_hubs_D1, rand_con_B_num_hubs_D1, con_B_num_hubs_D1, rand_con_A_num_hubs_D5, con_A_num_hubs_D5, rand_con_B_num_hubs_D5, con_B_num_hubs_D5, rand_con_A_num_hubs_D9, con_A_num_hubs_D9, rand_con_B_num_hubs_D9, con_B_num_hubs_D9]

#plt.figure(figsize=(20,10))
data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels, index=mouse_id_indices)
sns.swarmplot(data=data, color = 'k');
sns.boxplot(data=data, whis = 1.5, palette=my_pal);
# y, h, col = data['D9_B'].max() + 0.005, 0.005, 'k'
# x1, x2 = 2, 3  # columns to match
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)
# plt.text((x1+x2)*.5, y+h+0.01, "p = 0.024", ha='center', va='bottom', color=col)
plt.ylabel('Normalized hub count')
plt.title('Edge weight threshold: ' + str(threshold))
plt.show()
# plt.savefig('F31_hubs_fig.png', dpi = 300)

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
labels = ['D1_A', 'D1_B', 'D5_A', 'D5_B', 'D9_A', 'D9_B']
raw = [con_A_num_subnetworks_D1, con_B_num_subnetworks_D1, con_A_num_subnetworks_D5, con_B_num_subnetworks_D5, con_A_num_subnetworks_D9, con_B_num_subnetworks_D9]

# labels = ['rand_D1_A', 'D1_A', 'rand_D1_B', 'D1_B', 'rand_D5_A','D5_A', 'rand_D5_B','D5_B', 'rand_D9_A','D9_A', 'rand_D9_B', 'D9_B']
# raw = [rand_con_A_num_subnetworks_D1, con_A_num_subnetworks_D1, rand_con_B_num_subnetworks_D1,con_B_num_subnetworks_D1, rand_con_A_num_subnetworks_D5, con_A_num_subnetworks_D5, rand_con_B_num_subnetworks_D5, con_B_num_subnetworks_D5, rand_con_A_num_subnetworks_D9, con_A_num_subnetworks_D9, rand_con_B_num_subnetworks_D9, con_B_num_subnetworks_D9]

data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels, index=mouse_id_indices)

#fig, ax = plt.subplots(figsize=(20,10))
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
sns.boxplot(data=data, whis = 1.5, palette=my_pal);
sns.swarmplot(data=data, color = 'k');
# y, h, col = data['D1_A'].max() + 0.02, 0.01, 'k'
# x1, x2 = 0, 1  # columns to match
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
#
# plt.text((x1+x2)*.5, y+h+0.005, "p = 0.0625", ha='center', va='bottom', color=col)
plt.ylabel('Normalized subnetwork count')
plt.title('Edge weight threshold: ' + str(threshold))
plt.show()
#plt.savefig('F31_subnetwork_num_fig.png', dpi = 300)

#%% plot subnetwork size  metrics with seaborn 
labels = ['D1_A', 'D1_B', 'D5_A', 'D5_B', 'D9_A', 'D9_B']
raw = [con_A_size_subnetworks_D1, con_B_size_subnetworks_D1, con_A_size_subnetworks_D5, con_B_size_subnetworks_D5, con_A_size_subnetworks_D9, con_B_size_subnetworks_D9]

# labels = ['rand_D1_A', 'D1_A', 'rand_D1_B', 'D1_B', 'rand_D5_A','D5_A', 'rand_D5_B','D5_B', 'rand_D9_A','D9_A', 'rand_D9_B', 'D9_B']
# raw = [rand_con_A_size_subnetworks_D1, con_A_size_subnetworks_D1, rand_con_B_size_subnetworks_D1, con_B_size_subnetworks_D1, rand_con_A_size_subnetworks_D5, con_A_size_subnetworks_D5, rand_con_B_size_subnetworks_D5,con_B_size_subnetworks_D5, rand_con_A_size_subnetworks_D9, con_A_size_subnetworks_D9, rand_con_B_size_subnetworks_D9, con_B_size_subnetworks_D9]

data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels, index=mouse_id_indices)

#fig, ax = plt.subplots(figsize=(20,10))
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
sns.boxplot(data=data, whis = 1.5, palette=my_pal);
sns.swarmplot(data=data, color = 'k');
# y, h, col = data['D9_A'].max() + 0.05, 0.02, 'k'
# x1, x2 = 0, 2  # columns to match
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)
# plt.text((x1+x2)*.5, y+h+0.075, "p = 0.0034", ha='center', va='bottom', color=col)
plt.ylabel('Normalized average subnetwork size')
plt.title('Edge weight threshold: ' + str(threshold))
plt.show()
#plt.savefig('F31_subnetwork_size_fig.png', dpi = 300)

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

