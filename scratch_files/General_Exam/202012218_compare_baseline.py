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
day1_untreated = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv', '1055-3_D1_smoothed_calcium_traces.csv', '1055-4_D1_smoothed_calcium_traces.csv', '14-0_D1_smoothed_calcium_traces.csv']
day9_untreated = ['1055-1_D9_smoothed_calcium_traces.csv', '1055-2_D9_smoothed_calcium_traces.csv','1055-3_D9_smoothed_calcium_traces.csv', '1055-4_D9_smoothed_calcium_traces.csv', '14-0_D9_smoothed_calcium_traces.csv']
day0_untreated = ['1055-1_D0_smoothed_calcium_traces.csv','1055-2_D0_smoothed_calcium_traces.csv','1055-3_D0_smoothed_calcium_traces.csv','1055-4_D0_smoothed_calcium_traces.csv','14-0_D0_smoothed_calcium_traces.csv']
all_untreated_files = [day1_untreated, day9_untreated]

#%% Load treated data files - Th
day0_treated = ['348-1_D0_smoothed_calcium_traces.csv', '349-2_D0_smoothed_calcium_traces.csv', '386-2_D0_smoothed_calcium_traces.csv', '387-4_D0_smoothed_calcium_traces.csv', '396-1_D0_smoothed_calcium_traces.csv', '396-3_D0_smoothed_calcium_traces.csv']
day1_treated = ['348-1_D1_smoothed_calcium_traces.csv', '349-2_D1_smoothed_calcium_traces.csv', '386-2_D1_smoothed_calcium_traces.csv', '387-4_D1_smoothed_calcium_traces.csv', '396-1_D1_smoothed_calcium_traces.csv', '396-3_D1_smoothed_calcium_traces.csv']
day9_treated = ['348-1_D9_smoothed_calcium_traces.csv', '349-2_D9_smoothed_calcium_traces.csv', '386-2_D9_smoothed_calcium_traces.csv', '387-4_D9_smoothed_calcium_traces.csv', '396-1_D9_smoothed_calcium_traces.csv', '396-3_D9_smoothed_calcium_traces.csv']

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

con_A_cc_D1 = []
con_B_cc_D1 = []
con_A_cc_D9 = []
con_B_cc_D9 = []

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



        # hub analysis
        hubs_A, tmp = nn.get_context_A_hubs(threshold=threshold)
        hubs_B, tmp = nn.get_context_B_hubs(threshold=threshold)

        len_hubs_A = len(hubs_A)
        len_hubs_B = len(hubs_B)

        # clustering coefficient
        cc_A = nn.get_context_A_clustering_coefficient()
        cc_B = nn.get_context_B_clustering_coefficient()

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

#%% Baseline computation
subnetworks_D0 = []
num_subnetworks_D0 = []
size_subnetworks_D0 = []
hubs_D0 = []
num_hubs_D0 = []
edges_D0 = []
cc_D0 = []

for filename in day0_treated:
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

        # clustering coefficient
        cc_A = nn.get_context_A_clustering_coefficient()

        subnetworks_D0.append(connected_subnetworks)
        num_subnetworks_D0.append(num_connected_subnetworks / num_neurons)
        size_subnetworks_D0.append(np.median(len_connected_subnetworks) / num_neurons)
        edges_D0.append(list(G.edges()))
        hubs_D0.append(hubs)
        num_hubs_D0.append(len_hubs / num_neurons)
        cc_D0.append(np.average(cc_A))

#%% Plot baseline vs Day 1, 9 Hubs
baseline_inc = {"D0": "grey", "D1_A": "salmon", "D1_B": "darkturquoise", "D9_A":"salmon", "D9_B":"darkturquoise"}

labels = ['D0', 'D1_A', 'D1_B', 'D9_A', 'D9_B']
raw = [cc_D0, con_A_cc_D1, con_B_cc_D1, con_A_cc_D9, con_B_cc_D9]
data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels, index=mouse_id_indices)

sns.swarmplot(data=data, color = 'k');
sns.boxplot(data=data, whis = 1.5, palette=baseline_inc);

plt.ylabel('Normalized clustering coefficient')
plt.title('Edge weight threshold: ' + str(threshold))
plt.show()