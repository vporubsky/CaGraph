
from neuronal_network_graph import DGNetworkGraph as nng
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy import stats
import seaborn as sns
import os
import networkx as nx


# %% Load treated data files - Th
D0_Th = ['348-1_D0_smoothed_calcium_traces.csv',
         '349-2_D0_smoothed_calcium_traces.csv', '386-2_D0_smoothed_calcium_traces.csv',
         '387-4_D0_smoothed_calcium_traces.csv', '396-1_D0_smoothed_calcium_traces.csv',
         '396-3_D0_smoothed_calcium_traces.csv']
D1_Th = ['348-1_D1_smoothed_calcium_traces.csv',
         '349-2_D1_smoothed_calcium_traces.csv', '386-2_D1_smoothed_calcium_traces.csv',
         '387-4_D1_smoothed_calcium_traces.csv', '396-1_D1_smoothed_calcium_traces.csv',
         '396-3_D1_smoothed_calcium_traces.csv']
D5_Th = ['348-1_D5_smoothed_calcium_traces.csv',
         '349-2_D5_smoothed_calcium_traces.csv', '386-2_D5_smoothed_calcium_traces.csv',
         '387-4_D5_smoothed_calcium_traces.csv', '396-1_D5_smoothed_calcium_traces.csv',
         '396-3_D5_smoothed_calcium_traces.csv']
D9_Th = ['348-1_D9_smoothed_calcium_traces.csv',
         '349-2_D9_smoothed_calcium_traces.csv', '386-2_D9_smoothed_calcium_traces.csv',
         '387-4_D9_smoothed_calcium_traces.csv', '396-1_D9_smoothed_calcium_traces.csv',
         '396-3_D9_smoothed_calcium_traces.csv']

all_Th_files = [D0_Th, D1_Th, D5_Th, D9_Th]

#%% Load untreated data files - WT
D0_WT = ['1055-1_D0_smoothed_calcium_traces.csv','1055-2_D0_smoothed_calcium_traces.csv',
         '1055-3_D0_smoothed_calcium_traces.csv','1055-4_D0_smoothed_calcium_traces.csv',
         '14-0_D0_smoothed_calcium_traces.csv']
D1_WT = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv',
         '1055-3_D1_smoothed_calcium_traces.csv', '1055-4_D1_smoothed_calcium_traces.csv',
         '14-0_D1_smoothed_calcium_traces.csv', '122-1_D1_smoothed_calcium_traces.csv',
         '122-2_D1_smoothed_calcium_traces.csv', '122-3_D1_smoothed_calcium_traces.csv']
D5_WT = ['1055-1_D5_smoothed_calcium_traces.csv', '1055-2_D5_smoothed_calcium_traces.csv',
         '1055-3_D5_smoothed_calcium_traces.csv', '1055-4_D5_smoothed_calcium_traces.csv',
         '14-0_D5_smoothed_calcium_traces.csv']
D9_WT = ['1055-1_D9_smoothed_calcium_traces.csv', '1055-2_D9_smoothed_calcium_traces.csv',
         '1055-3_D9_smoothed_calcium_traces.csv', '1055-4_D9_smoothed_calcium_traces.csv',
         '14-0_D9_smoothed_calcium_traces.csv', '122-1_D9_smoothed_calcium_traces.csv',
         '122-2_D9_smoothed_calcium_traces.csv', '122-3_D9_smoothed_calcium_traces.csv']

all_WT_files = [D0_WT, D1_WT, D5_WT, D9_WT]


# %% All measurements, separating contexts
threshold = 0.3
names = []
data_mat = []

WT_con_A_subnetworks_D0 = []
WT_con_B_subnetworks_D0 = []
WT_con_A_subnetworks_D1 = []
WT_con_B_subnetworks_D1 = []
WT_con_A_subnetworks_D5 = []
WT_con_B_subnetworks_D5 = []
WT_con_A_subnetworks_D9 = []
WT_con_B_subnetworks_D9 = []

WT_con_A_num_subnetworks_D0 = []
WT_con_B_num_subnetworks_D0 = []
WT_con_A_num_subnetworks_D1 = []
WT_con_B_num_subnetworks_D1 = []
WT_con_A_num_subnetworks_D5 = []
WT_con_B_num_subnetworks_D5 = []
WT_con_A_num_subnetworks_D9 = []
WT_con_B_num_subnetworks_D9 = []

WT_con_A_size_subnetworks_D0 = []
WT_con_B_size_subnetworks_D0 = []
WT_con_A_size_subnetworks_D1 = []
WT_con_B_size_subnetworks_D1 = []
WT_con_A_size_subnetworks_D5 = []
WT_con_B_size_subnetworks_D5 = []
WT_con_A_size_subnetworks_D9 = []
WT_con_B_size_subnetworks_D9 = []

WT_con_A_cc_D0 = []
WT_con_B_cc_D0 = []
WT_con_A_cc_D1 = []
WT_con_B_cc_D1 = []
WT_con_A_cc_D5 = []
WT_con_B_cc_D5 = []
WT_con_A_cc_D9 = []
WT_con_B_cc_D9 = []

WT_con_A_hubs_D0 = []
WT_con_B_hubs_D0 = []
WT_con_A_hubs_D1 = []
WT_con_B_hubs_D1 = []
WT_con_A_hubs_D5 = []
WT_con_B_hubs_D5 = []
WT_con_A_hubs_D9 = []
WT_con_B_hubs_D9 = []

WT_con_A_num_hubs_D0 = []
WT_con_B_num_hubs_D0 = []
WT_con_A_num_hubs_D1 = []
WT_con_B_num_hubs_D1 = []
WT_con_A_num_hubs_D5 = []
WT_con_B_num_hubs_D5 = []
WT_con_A_num_hubs_D9 = []
WT_con_B_num_hubs_D9 = []

WT_con_A_hits_D0 = []
WT_con_B_hits_D0 = []
WT_con_A_hits_D1 = []
WT_con_B_hits_D1 = []
WT_con_A_hits_D5 = []
WT_con_B_hits_D5 = []
WT_con_A_hits_D9 = []
WT_con_B_hits_D9 = []

WT_con_A_cr_D0 = []
WT_con_B_cr_D0 = []
WT_con_A_cr_D1 = []
WT_con_B_cr_D1 = []
WT_con_A_cr_D5 = []
WT_con_B_cr_D5 = []
WT_con_A_cr_D9 = []
WT_con_B_cr_D9 = []

WT_edges_A_D0 = []
WT_edges_B_D0 = []
WT_edges_A_D1 = []
WT_edges_B_D1 = []
WT_edges_A_D5 = []
WT_edges_B_D5 = []
WT_edges_A_D9 = []
WT_edges_B_D9 = []

mouse_id_indices = []

# %% Context A and B with WT subjects
# Loop through all subjects and perform experimental and randomized network analyses
for day in [0, 1, 2, 3]:
    for mouse_id_index in range(len(all_WT_files[day])):
        filename = all_WT_files[day][mouse_id_index]
        mouse_id = filename.strip('_smoothed_calcium_traces.csv')

        if day == 0:
            mouse_id_indices.append(mouse_id.replace('_D0', ''))

        nn = nng(filename)
        print(f"Executing analyses for {mouse_id}")
        num_neurons = nn.num_neurons

        # Context A and B graphs
        conA = nn.get_context_A_graph(threshold=threshold)
        conB = nn.get_context_B_graph(threshold=threshold)

        # subnetwork analysis
        connected_subnetworks_A = nn.get_context_A_subnetworks(threshold=threshold)
        connected_subnetworks_B = nn.get_context_B_subnetworks(threshold=threshold)

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
        hubs_A, hits_A = nn.get_context_A_hubs(threshold=threshold)
        hubs_B, hits_B = nn.get_context_B_hubs(threshold=threshold)

        len_hubs_A = len(hubs_A)
        len_hubs_B = len(hubs_B)

        # correlated pairs ratio
        cr_A = nn.get_context_A_correlated_pair_ratio(threshold=threshold)
        cr_B = nn.get_context_B_correlated_pair_ratio(threshold=threshold)

        # communities
        # community_A = list(nx.algorithms.community.greedy_modularity_communities(conA))
        # community_B = list(nx.algorithms.community.greedy_modularity_communities(conB))

        if day == 0:
            WT_con_A_subnetworks_D0.append(connected_subnetworks_A)
            WT_con_B_subnetworks_D0.append(connected_subnetworks_B)
            WT_con_A_num_subnetworks_D0.append(num_connected_subnetworks_A / num_neurons)
            WT_con_B_num_subnetworks_D0.append(num_connected_subnetworks_B / num_neurons)
            WT_con_A_size_subnetworks_D0.append(np.median(len_connected_subnetworks_A) / num_neurons)
            WT_con_B_size_subnetworks_D0.append(np.median(len_connected_subnetworks_B) / num_neurons)
            WT_edges_A_D0.append(list(conA.edges()))
            WT_edges_B_D0.append(list(conB.edges()))
            WT_con_A_hubs_D0.append(hubs_A)
            WT_con_B_hubs_D0.append(hubs_B)
            WT_con_A_num_hubs_D0.append(len_hubs_A / num_neurons)
            WT_con_B_num_hubs_D0.append(len_hubs_B / num_neurons)
            WT_con_A_cc_D0.append(cc_A)
            WT_con_B_cc_D0.append(cc_B)
            WT_con_A_hits_D0.append([hit_val/num_neurons for hit_val in list(hits_A.values())])
            WT_con_B_hits_D0.append([hit_val/num_neurons for hit_val in list(hits_B.values())])
            WT_con_A_cr_D0.append(cr_A)
            WT_con_B_cr_D0.append(cr_B)

        elif day == 1:
            WT_con_A_subnetworks_D1.append(connected_subnetworks_A)
            WT_con_B_subnetworks_D1.append(connected_subnetworks_B)
            WT_con_A_num_subnetworks_D1.append(num_connected_subnetworks_A / num_neurons)
            WT_con_B_num_subnetworks_D1.append(num_connected_subnetworks_B / num_neurons)
            WT_con_A_size_subnetworks_D1.append(np.median(len_connected_subnetworks_A) / num_neurons)
            WT_con_B_size_subnetworks_D1.append(np.median(len_connected_subnetworks_B) / num_neurons)
            WT_edges_A_D1.append(list(conA.edges()))
            WT_edges_B_D1.append(list(conB.edges()))
            WT_con_A_hubs_D1.append(hubs_A)
            WT_con_B_hubs_D1.append(hubs_B)
            WT_con_A_num_hubs_D1.append(len_hubs_A / num_neurons)
            WT_con_B_num_hubs_D1.append(len_hubs_B / num_neurons)
            WT_con_A_cc_D1.append(cc_A)
            WT_con_B_cc_D1.append(cc_B)
            WT_con_A_hits_D1.append(list(hits_A.values()))
            WT_con_B_hits_D1.append(list(hits_B.values()))
            WT_con_A_cr_D1.append(cr_A)
            WT_con_B_cr_D1.append(cr_B)

        elif day == 2:
            WT_con_A_subnetworks_D5.append(connected_subnetworks_A)
            WT_con_B_subnetworks_D5.append(connected_subnetworks_B)
            WT_con_A_num_subnetworks_D5.append(num_connected_subnetworks_A / num_neurons)
            WT_con_B_num_subnetworks_D5.append(num_connected_subnetworks_B / num_neurons)
            WT_con_A_size_subnetworks_D5.append(np.median(len_connected_subnetworks_A) / num_neurons)
            WT_con_B_size_subnetworks_D5.append(np.median(len_connected_subnetworks_B) / num_neurons)
            WT_edges_A_D5.append(list(conA.edges()))
            WT_edges_B_D5.append(list(conB.edges()))
            WT_con_A_hubs_D5.append(hubs_A)
            WT_con_B_hubs_D5.append(hubs_B)
            WT_con_A_num_hubs_D5.append(len_hubs_A / num_neurons)
            WT_con_B_num_hubs_D5.append(len_hubs_B / num_neurons)
            WT_con_A_cc_D5.append(cc_A)
            WT_con_B_cc_D5.append(cc_B)
            WT_con_A_hits_D5.append([hit_val/num_neurons for hit_val in list(hits_A.values())])
            WT_con_B_hits_D5.append([hit_val/num_neurons for hit_val in list(hits_B.values())])
            WT_con_A_cr_D5.append(cr_A)
            WT_con_B_cr_D5.append(cr_B)

        elif day == 3:
            WT_con_A_subnetworks_D9.append(connected_subnetworks_A)
            WT_con_B_subnetworks_D9.append(connected_subnetworks_B)
            WT_con_A_num_subnetworks_D9.append(num_connected_subnetworks_A / num_neurons)
            WT_con_B_num_subnetworks_D9.append(num_connected_subnetworks_B / num_neurons)
            WT_con_A_size_subnetworks_D9.append(np.median(len_connected_subnetworks_A) / num_neurons)
            WT_con_B_size_subnetworks_D9.append(np.median(len_connected_subnetworks_B) / num_neurons)
            WT_edges_A_D9.append(list(conA.edges()))
            WT_edges_B_D9.append(list(conB.edges()))
            WT_con_A_hubs_D9.append(hubs_A)
            WT_con_B_hubs_D9.append(hubs_B)
            WT_con_A_num_hubs_D9.append(len_hubs_A / num_neurons)
            WT_con_B_num_hubs_D9.append(len_hubs_B / num_neurons)
            WT_con_A_cc_D9.append(cc_A)
            WT_con_B_cc_D9.append(cc_B)
            WT_con_A_hits_D9.append([hit_val/num_neurons for hit_val in list(hits_A.values())])
            WT_con_B_hits_D9.append([hit_val/num_neurons for hit_val in list(hits_B.values())])
            WT_con_A_cr_D9.append(cr_A)
            WT_con_B_cr_D9.append(cr_B)


# %% All measurements, separating contexts
threshold = 0.3
names = []
data_mat = []

Th_con_A_subnetworks_D0 = []
Th_con_B_subnetworks_D0 = []
Th_con_A_subnetworks_D1 = []
Th_con_B_subnetworks_D1 = []
Th_con_A_subnetworks_D5 = []
Th_con_B_subnetworks_D5 = []
Th_con_A_subnetworks_D9 = []
Th_con_B_subnetworks_D9 = []

Th_con_A_num_subnetworks_D0 = []
Th_con_B_num_subnetworks_D0 = []
Th_con_A_num_subnetworks_D1 = []
Th_con_B_num_subnetworks_D1 = []
Th_con_A_num_subnetworks_D5 = []
Th_con_B_num_subnetworks_D5 = []
Th_con_A_num_subnetworks_D9 = []
Th_con_B_num_subnetworks_D9 = []

Th_con_A_size_subnetworks_D0 = []
Th_con_B_size_subnetworks_D0 = []
Th_con_A_size_subnetworks_D1 = []
Th_con_B_size_subnetworks_D1 = []
Th_con_A_size_subnetworks_D5 = []
Th_con_B_size_subnetworks_D5 = []
Th_con_A_size_subnetworks_D9 = []
Th_con_B_size_subnetworks_D9 = []

Th_con_A_cc_D0 = []
Th_con_B_cc_D0 = []
Th_con_A_cc_D1 = []
Th_con_B_cc_D1 = []
Th_con_A_cc_D5 = []
Th_con_B_cc_D5 = []
Th_con_A_cc_D9 = []
Th_con_B_cc_D9 = []

Th_con_A_hubs_D0 = []
Th_con_B_hubs_D0 = []
Th_con_A_hubs_D1 = []
Th_con_B_hubs_D1 = []
Th_con_A_hubs_D5 = []
Th_con_B_hubs_D5 = []
Th_con_A_hubs_D9 = []
Th_con_B_hubs_D9 = []

Th_con_A_num_hubs_D0 = []
Th_con_B_num_hubs_D0 = []
Th_con_A_num_hubs_D1 = []
Th_con_B_num_hubs_D1 = []
Th_con_A_num_hubs_D5 = []
Th_con_B_num_hubs_D5 = []
Th_con_A_num_hubs_D9 = []
Th_con_B_num_hubs_D9 = []

Th_con_A_hits_D0 = []
Th_con_B_hits_D0 = []
Th_con_A_hits_D1 = []
Th_con_B_hits_D1 = []
Th_con_A_hits_D5 = []
Th_con_B_hits_D5 = []
Th_con_A_hits_D9 = []
Th_con_B_hits_D9 = []

Th_con_A_cr_D0 = []
Th_con_B_cr_D0 = []
Th_con_A_cr_D1 = []
Th_con_B_cr_D1 = []
Th_con_A_cr_D5 = []
Th_con_B_cr_D5 = []
Th_con_A_cr_D9 = []
Th_con_B_cr_D9 = []

Th_edges_A_D0 = []
Th_edges_B_D0 = []
Th_edges_A_D1 = []
Th_edges_B_D1 = []
Th_edges_A_D5 = []
Th_edges_B_D5 = []
Th_edges_A_D9 = []
Th_edges_B_D9 = []

mouse_id_indices = []

# %% Context A and B
# Loop through all subjects and perform experimental and randomized network analyses
for day in [0, 1, 2, 3]:
    for mouse_id_index in range(len(all_Th_files[day])):
        filename = all_Th_files[day][mouse_id_index]
        mouse_id = filename.strip('_smoothed_calcium_traces.csv')

        if day == 0:
            mouse_id_indices.append(mouse_id.replace('_D0', ''))

        nn = nng(filename)
        print(f"Executing analyses for {mouse_id}")
        num_neurons = nn.num_neurons

        # Context A and B graphs
        conA = nn.get_context_A_graph(threshold=threshold)
        conB = nn.get_context_B_graph(threshold=threshold)

        # subnetwork analysis
        connected_subnetworks_A = nn.get_context_A_subnetworks(threshold=threshold)
        connected_subnetworks_B = nn.get_context_B_subnetworks(threshold=threshold)

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
        hubs_A, hits_A = nn.get_context_A_hubs(threshold=threshold)
        hubs_B, hits_B = nn.get_context_B_hubs(threshold=threshold)

        len_hubs_A = len(hubs_A)
        len_hubs_B = len(hubs_B)


        if day == 0:
            Th_con_A_subnetworks_D0.append(connected_subnetworks_A)
            Th_con_B_subnetworks_D0.append(connected_subnetworks_B)
            Th_con_A_num_subnetworks_D0.append(num_connected_subnetworks_A / num_neurons)
            Th_con_B_num_subnetworks_D0.append(num_connected_subnetworks_B / num_neurons)
            Th_con_A_size_subnetworks_D0.append(np.median(len_connected_subnetworks_A) / num_neurons)
            Th_con_B_size_subnetworks_D0.append(np.median(len_connected_subnetworks_B) / num_neurons)
            Th_edges_A_D0.append(list(conA.edges()))
            Th_edges_B_D0.append(list(conB.edges()))
            Th_con_A_hubs_D0.append(hubs_A)
            Th_con_B_hubs_D0.append(hubs_B)
            Th_con_A_num_hubs_D0.append(len_hubs_A / num_neurons)
            Th_con_B_num_hubs_D0.append(len_hubs_B / num_neurons)
            Th_con_A_cc_D0.append(cc_A)
            Th_con_B_cc_D0.append(cc_B)

        elif day == 1:
            Th_con_A_subnetworks_D1.append(connected_subnetworks_A)
            Th_con_B_subnetworks_D1.append(connected_subnetworks_B)
            Th_con_A_num_subnetworks_D1.append(num_connected_subnetworks_A / num_neurons)
            Th_con_B_num_subnetworks_D1.append(num_connected_subnetworks_B / num_neurons)
            Th_con_A_size_subnetworks_D1.append(np.median(len_connected_subnetworks_A) / num_neurons)
            Th_con_B_size_subnetworks_D1.append(np.median(len_connected_subnetworks_B) / num_neurons)
            Th_edges_A_D1.append(list(conA.edges()))
            Th_edges_B_D1.append(list(conB.edges()))
            Th_con_A_hubs_D1.append(hubs_A)
            Th_con_B_hubs_D1.append(hubs_B)
            Th_con_A_num_hubs_D1.append(len_hubs_A / num_neurons)
            Th_con_B_num_hubs_D1.append(len_hubs_B / num_neurons)
            Th_con_A_cc_D1.append(cc_A)
            Th_con_B_cc_D1.append(cc_B)


        elif day == 2:
            Th_con_A_subnetworks_D5.append(connected_subnetworks_A)
            Th_con_B_subnetworks_D5.append(connected_subnetworks_B)
            Th_con_A_num_subnetworks_D5.append(num_connected_subnetworks_A / num_neurons)
            Th_con_B_num_subnetworks_D5.append(num_connected_subnetworks_B / num_neurons)
            Th_con_A_size_subnetworks_D5.append(np.median(len_connected_subnetworks_A) / num_neurons)
            Th_con_B_size_subnetworks_D5.append(np.median(len_connected_subnetworks_B) / num_neurons)
            Th_edges_A_D5.append(list(conA.edges()))
            Th_edges_B_D5.append(list(conB.edges()))
            Th_con_A_hubs_D5.append(hubs_A)
            Th_con_B_hubs_D5.append(hubs_B)
            Th_con_A_num_hubs_D5.append(len_hubs_A / num_neurons)
            Th_con_B_num_hubs_D5.append(len_hubs_B / num_neurons)
            Th_con_A_cc_D5.append(cc_A)
            Th_con_B_cc_D5.append(cc_B)


        elif day == 3:
            Th_con_A_subnetworks_D9.append(connected_subnetworks_A)
            Th_con_B_subnetworks_D9.append(connected_subnetworks_B)
            Th_con_A_num_subnetworks_D9.append(num_connected_subnetworks_A / num_neurons)
            Th_con_B_num_subnetworks_D9.append(num_connected_subnetworks_B / num_neurons)
            Th_con_A_size_subnetworks_D9.append(np.median(len_connected_subnetworks_A) / num_neurons)
            Th_con_B_size_subnetworks_D9.append(np.median(len_connected_subnetworks_B) / num_neurons)
            Th_edges_A_D9.append(list(conA.edges()))
            Th_edges_B_D9.append(list(conB.edges()))
            Th_con_A_hubs_D9.append(hubs_A)
            Th_con_B_hubs_D9.append(hubs_B)
            Th_con_A_num_hubs_D9.append(len_hubs_A / num_neurons)
            Th_con_B_num_hubs_D9.append(len_hubs_B / num_neurons)
            Th_con_A_cc_D9.append(cc_A)
            Th_con_B_cc_D9.append(cc_B)

#%%
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
from scipy import stats

idx=1
plt.figure(figsize=(15,10))
plt.subplot(221)
sns.distplot(WT_con_A_cc_D1[idx], hist=True, kde=True,
             bins=int(25), color = 'mistyrose',
             hist_kws={'edgecolor':'mistyrose'},
             kde_kws={'linewidth': 2})
sns.distplot(Th_con_A_cc_D1[idx], hist=True, kde=True,
             bins=int(25), color = 'salmon',
             hist_kws={'edgecolor':'salmon'},
             kde_kws={"color": "salmon",'linewidth': 2})
plt.xlabel('')
plt.ylabel('')

stat_lev = stats.ks_2samp(WT_con_A_cc_D1[idx], Th_con_A_cc_D1[idx])
plt.text(0.2, 5, f'KS P-value: {stat_lev.pvalue:.2e}')


plt.subplot(222)
sns.distplot(WT_con_B_cc_D1[idx], hist=True, kde=True,
             bins=int(25), color = 'turquoise',
             hist_kws={'edgecolor':'turquoise'},
             kde_kws={'linewidth': 2})
sns.distplot(Th_con_B_cc_D1[idx], hist=True, kde=True,
             bins=int(25), color = 'teal',
             hist_kws={'edgecolor':'teal'},
             kde_kws={"color": "teal", 'linewidth': 2})

plt.xlabel('')
plt.ylabel('')
stat_lev = stats.ks_2samp(WT_con_B_cc_D1[idx], Th_con_B_cc_D1[idx])
plt.text(0.2, 5, f'KS P-value: {stat_lev.pvalue:.2e}')


plt.subplot(223)
sns.distplot(WT_con_A_cc_D9[idx], hist=True, kde=True,
             bins=int(25), color = 'mistyrose',
             hist_kws={'edgecolor':'mistyrose'},
             kde_kws={'linewidth': 2})
sns.distplot(Th_con_A_cc_D9[idx], hist=True, kde=True,
             bins=int(25), color = 'salmon',
             hist_kws={'edgecolor':'salmon'},
             kde_kws={"color": "salmon",'linewidth': 2})

plt.xlabel('')
plt.ylabel('')

stat_lev = stats.ks_2samp(WT_con_A_cc_D9[idx], Th_con_A_cc_D9[idx])
plt.text(0.2, 5, f'KS P-value: {stat_lev.pvalue:.2e}')

plt.subplot(224)
sns.distplot(WT_con_B_cc_D9[idx], hist=True, kde=True,
             bins=int(25), color = 'turquoise',
             hist_kws={'edgecolor':'turquoise'},
             kde_kws={'linewidth': 2})
sns.distplot(Th_con_B_cc_D9[idx], hist=True, kde=True,
             bins=int(25), color = 'teal',
             hist_kws={'edgecolor':'teal'},
             kde_kws={"color": "teal", 'linewidth': 2})

plt.yticks([])
plt.xlabel('')
plt.ylabel('')

stat_lev = stats.ks_2samp(WT_con_B_cc_D9[idx], Th_con_B_cc_D9[idx])
plt.text(0.2, 5, f'KS P-value: {stat_lev.pvalue:.2e}')
#plt.savefig(os.path.join(os.getcwd(), f"NIH_F31/clustering_coefficient.svg"), dpi=300, transparent=True)
plt.show()



#%%
idx = 4
plt.figure(figsize=(15,10))
plt.subplot(221)
sns.distplot(WT_con_A_cc_D1[idx], hist=True, kde=True,
             bins=int(25), color = 'mistyrose',
             hist_kws={'edgecolor':'mistyrose'},
             kde_kws={'linewidth': 2})
sns.distplot(WT_con_B_cc_D1[idx], hist=True, kde=True,
             bins=int(25), color = 'turquoise',
             hist_kws={'edgecolor':'turquoise'},
             kde_kws={"color": "turquoise",'linewidth': 2})
plt.xlabel('')
plt.ylabel('')

stat_lev = stats.ks_2samp(WT_con_A_cc_D1[idx], WT_con_B_cc_D1[idx])
plt.text(0.2, 5, f'KS P-value: {stat_lev.pvalue:.2e}')


plt.subplot(222)
sns.distplot(WT_con_A_cc_D9[idx], hist=True, kde=True,
             bins=int(25), color = 'mistyrose',
             hist_kws={'edgecolor':'mistyrose'},
             kde_kws={'linewidth': 2})
sns.distplot(WT_con_B_cc_D9[idx], hist=True, kde=True,
             bins=int(25), color = 'turquoise',
             hist_kws={'edgecolor':'turquoise'},
             kde_kws={"color": "turquoise",'linewidth': 2})
plt.xlabel('')
plt.ylabel('')

stat_lev = stats.ks_2samp(WT_con_A_cc_D9[idx], WT_con_B_cc_D9[idx])
plt.text(0.2, 5, f'KS P-value: {stat_lev.pvalue:.2e}')


plt.subplot(223)
sns.distplot(Th_con_A_cc_D1[idx], hist=True, kde=True,
             bins=int(25), color = 'salmon',
             hist_kws={'edgecolor':'salmon'},
             kde_kws={'linewidth': 2})
sns.distplot(Th_con_B_cc_D1[idx], hist=True, kde=True,
             bins=int(25), color = 'teal',
             hist_kws={'edgecolor':'teal'},
             kde_kws={"color": "teal",'linewidth': 2})
plt.xlabel('')
plt.ylabel('')

stat_lev = stats.ks_2samp(Th_con_A_cc_D1[idx], Th_con_B_cc_D1[idx])
plt.text(0.2, 5, f'KS P-value: {stat_lev.pvalue:.2e}')


plt.subplot(224)
sns.distplot(Th_con_A_cc_D9[idx], hist=True, kde=True,
             bins=int(25), color = 'salmon',
             hist_kws={'edgecolor':'salmon'},
             kde_kws={'linewidth': 2})
sns.distplot(Th_con_B_cc_D9[idx], hist=True, kde=True,
             bins=int(25), color = 'teal',
             hist_kws={'edgecolor':'teal'},
             kde_kws={"color": "teal",'linewidth': 2})
plt.xlabel('')
plt.ylabel('')

stat_lev = stats.ks_2samp(Th_con_A_cc_D9[idx], Th_con_B_cc_D9[idx])
plt.text(0.2, 5, f'KS P-value: {stat_lev.pvalue:.2e}')
#plt.savefig(os.path.join(os.getcwd(), f"NIH_F31/clustering_coefficient.svg"), dpi=300, transparent=True)
plt.show()


#%%
#%% Clustering coefficeint
# WT correlated pairs ratio
WT_con_A_cc_D0 = [item for sublist in WT_con_A_cc_D0 for item in sublist]
WT_con_B_cc_D0 = [item for sublist in WT_con_B_cc_D0 for item in sublist]

WT_con_A_cc_D1 = [item for sublist in WT_con_A_cc_D1 for item in sublist]
WT_con_B_cc_D1 = [item for sublist in WT_con_B_cc_D1 for item in sublist]

WT_con_A_cc_D5 = [item for sublist in WT_con_A_cc_D5 for item in sublist]
WT_con_B_cc_D5 = [item for sublist in WT_con_B_cc_D5 for item in sublist]

WT_con_A_cc_D9 = [item for sublist in WT_con_A_cc_D9 for item in sublist]
WT_con_B_cc_D9 = [item for sublist in WT_con_B_cc_D9 for item in sublist]

Th_con_A_cc_D0 = [item for sublist in Th_con_A_cc_D0 for item in sublist]
Th_con_B_cc_D0 = [item for sublist in Th_con_B_cc_D0 for item in sublist]

# Th correlated pairs ratio
Th_con_A_cc_D1 = [item for sublist in Th_con_A_cc_D1 for item in sublist]
Th_con_B_cc_D1 = [item for sublist in Th_con_B_cc_D1 for item in sublist]

Th_con_A_cc_D5 = [item for sublist in Th_con_A_cc_D5 for item in sublist]
Th_con_B_cc_D5 = [item for sublist in Th_con_B_cc_D5 for item in sublist]

Th_con_A_cc_D9 = [item for sublist in Th_con_A_cc_D9 for item in sublist]
Th_con_B_cc_D9 = [item for sublist in Th_con_B_cc_D9 for item in sublist]


#%% Context B data

linewidth_style=3
plt.figure(figsize=(15,15))
# Plot
plt.subplot(221)

plt.boxplot([WT_con_A_cc_D1, Th_con_A_cc_D1], positions=[0, 1], capprops=dict(linewidth=linewidth_style, color='salmon'),
           whiskerprops=dict(linewidth=linewidth_style, color='salmon'),
           boxprops=dict(linewidth=linewidth_style, color='salmon'),
           medianprops=dict(linewidth=linewidth_style, color='salmon'))
plt.xlabel('')
plt.ylabel('')
plt.xticks([])

plt.subplot(222)

plt.boxplot([WT_con_B_cc_D1, Th_con_B_cc_D1], positions=[0, 1], capprops=dict(linewidth=linewidth_style, color='turquoise'),
           whiskerprops=dict(linewidth=linewidth_style, color='turquoise'),
           boxprops=dict(linewidth=linewidth_style, color='turquoise'),
           medianprops=dict(linewidth=linewidth_style, color='turquoise'))
plt.xlabel('')
plt.ylabel('')
plt.xticks([])

plt.subplot(223)
plt.boxplot([WT_con_A_cc_D9, Th_con_A_cc_D9], positions=[0, 1], capprops=dict(linewidth=linewidth_style, color='salmon'),
           whiskerprops=dict(linewidth=linewidth_style, color='salmon'),
           boxprops=dict(linewidth=linewidth_style, color='salmon'),
           medianprops=dict(linewidth=linewidth_style, color='salmon'))
plt.xlabel('')
plt.ylabel('')
plt.xticks([])

plt.subplot(224)

plt.boxplot([WT_con_B_cc_D9, Th_con_B_cc_D9], positions=[0, 1], capprops=dict(linewidth=linewidth_style, color='turquoise'),
           whiskerprops=dict(linewidth=linewidth_style, color='turquoise'),
           boxprops=dict(linewidth=linewidth_style, color='turquoise'),
           medianprops=dict(linewidth=linewidth_style, color='turquoise'))
plt.xlabel('')
plt.ylabel('')
plt.xticks([])

plt.show()

#%%

linewidth_style=3
plt.figure(figsize=(15,15))
# Plot
plt.subplot(221)

plt.boxplot([WT_con_A_cc_D1, WT_con_B_cc_D1], positions=[0, 1], capprops=dict(linewidth=linewidth_style, color='salmon'),
           whiskerprops=dict(linewidth=linewidth_style, color='salmon'),
           boxprops=dict(linewidth=linewidth_style, color='salmon'),
           medianprops=dict(linewidth=linewidth_style, color='salmon'))
plt.xlabel('')
plt.ylabel('')
plt.xticks([])
plt.ylim(0,1)

plt.subplot(222)

plt.boxplot([WT_con_A_cc_D9, WT_con_B_cc_D9], positions=[0, 1], capprops=dict(linewidth=linewidth_style, color='turquoise'),
           whiskerprops=dict(linewidth=linewidth_style, color='turquoise'),
           boxprops=dict(linewidth=linewidth_style, color='turquoise'),
           medianprops=dict(linewidth=linewidth_style, color='turquoise'))
plt.xlabel('')
plt.ylabel('')
plt.xticks([])
plt.ylim(0,1)

plt.subplot(223)
plt.boxplot([Th_con_A_cc_D1, Th_con_B_cc_D1], positions=[0, 1], capprops=dict(linewidth=linewidth_style, color='salmon'),
           whiskerprops=dict(linewidth=linewidth_style, color='salmon'),
           boxprops=dict(linewidth=linewidth_style, color='salmon'),
           medianprops=dict(linewidth=linewidth_style, color='salmon'))
plt.xlabel('')
plt.ylabel('')
plt.xticks([])
plt.ylim(0,1)

plt.subplot(224)

plt.boxplot([Th_con_A_cc_D9, Th_con_B_cc_D9], positions=[0, 1], capprops=dict(linewidth=linewidth_style, color='turquoise'),
           whiskerprops=dict(linewidth=linewidth_style, color='turquoise'),
           boxprops=dict(linewidth=linewidth_style, color='turquoise'),
           medianprops=dict(linewidth=linewidth_style, color='turquoise'))
plt.xlabel('')
plt.ylabel('')
plt.xticks([])
plt.ylim(0,1)

plt.show()


#%%
plt.figure(figsize=(15,15))
import seaborn as sns

plt.subplot(222)
ax = sns.violinplot(np.zeros(np.shape(WT_con_B_cc_D1)), WT_con_B_cc_D1,color="turquoise", cut=0)
sns.violinplot(np.ones(np.shape(Th_con_B_cc_D1)), Th_con_B_cc_D1,color="teal", cut=0)
plt.setp(ax.collections, alpha=.3)
plt.yticks([])
plt.xticks([])
plt.ylim(0,1)
plt.subplot(224)
ax = sns.violinplot(np.zeros(np.shape(WT_con_B_cc_D9)), WT_con_B_cc_D9,color="turquoise", cut=0)
sns.violinplot(np.ones(np.shape(Th_con_B_cc_D9)), Th_con_B_cc_D9,color="teal", cut=0)
plt.setp(ax.collections, alpha=.3)
plt.yticks([])
plt.xticks([])
plt.ylim(0,1)
plt.subplot(221)
ax = sns.violinplot(np.zeros(np.shape(WT_con_A_cc_D1)), WT_con_A_cc_D1,color="mistyrose", cut=0)
sns.violinplot(np.ones(np.shape(Th_con_A_cc_D1)), Th_con_A_cc_D1,color="salmon", cut=0)
plt.setp(ax.collections, alpha=.3)
plt.yticks([0.5, 1])
plt.xticks([])
plt.ylim(0,1)
plt.subplot(223)
ax = sns.violinplot(np.zeros(np.shape(WT_con_A_cc_D9)), WT_con_A_cc_D9,color="mistyrose", cut=0)
sns.violinplot(np.ones(np.shape(Th_con_A_cc_D9)), Th_con_A_cc_D9,color="salmon", cut=0)
plt.setp(ax.collections, alpha=.3)
plt.yticks([0.5, 1])
plt.xticks([])
plt.ylim(0,1)
plt.show()