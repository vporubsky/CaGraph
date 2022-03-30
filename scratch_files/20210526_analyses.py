"""
Plotting the cumulative density function (CDF) for each individual mouse
to identify notable outlier subjects.

# Todo: Add some metric to actually pick out outliers automatically based on statistics.
"""
from neuronal_network_graph import DGNetworkGraph as nng
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os


#%% Global analysis parameters
threshold = 0.3

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

# %% All measurements, separating contexts
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

WT_mouse_id_indices = []

# %% Context A and B with WT subjects
# Loop through all subjects and perform experimental and randomized network analyses
for day in [0, 1, 2, 3]:
    for mouse_id_index in range(len(all_WT_files[day])):
        filename = all_WT_files[day][mouse_id_index]
        mouse_id = filename.strip('_smoothed_calcium_traces.csv')

        if day == 0:
            WT_mouse_id_indices.append(mouse_id.replace('_D0', ''))

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

Th_mouse_id_indices = []

# %% Context A and B
# Loop through all subjects and perform experimental and randomized network analyses
for day in [0, 1, 2, 3]:
    for mouse_id_index in range(len(all_Th_files[day])):
        filename = all_Th_files[day][mouse_id_index]
        mouse_id = filename.strip('_smoothed_calcium_traces.csv')

        if day == 0:
            Th_mouse_id_indices.append(mouse_id.replace('_D0', ''))

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
            Th_con_A_hits_D0.append([hit_val/num_neurons for hit_val in list(hits_A.values())])
            Th_con_B_hits_D0.append([hit_val/num_neurons for hit_val in list(hits_B.values())])
            Th_con_A_cr_D0.append(cr_A)
            Th_con_B_cr_D0.append(cr_B)

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
            Th_con_A_hits_D1.append([hit_val/num_neurons for hit_val in list(hits_A.values())])
            Th_con_B_hits_D1.append([hit_val/num_neurons for hit_val in list(hits_B.values())])
            Th_con_A_cr_D1.append(cr_A)
            Th_con_B_cr_D1.append(cr_B)

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
            Th_con_A_hits_D5.append(list(hits_A.values()))
            Th_con_B_hits_D5.append(list(hits_B.values()))
            Th_con_A_cr_D5.append(cr_A)
            Th_con_B_cr_D5.append(cr_B)

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
            Th_con_A_hits_D9.append(list(hits_A.values()))
            Th_con_B_hits_D9.append(list(hits_B.values()))
            Th_con_A_cr_D9.append(cr_A)
            Th_con_B_cr_D9.append(cr_B)


#%% WT Correlated Pairs Plots: con A v con B
for idx in range(len(WT_mouse_id_indices)):
    # CDF prestim, stim ----------------------------------------------
    stat_lev = stats.ks_2samp(WT_con_A_cr_D1[idx], WT_con_B_cr_D1[idx])

    # sort the data in ascending order
    x = np.sort(WT_con_A_cr_D1[idx])
    # get the cdf values of y
    y = np.arange(len(WT_con_A_cr_D1[idx])) / float(len(WT_con_A_cr_D1[idx]))

    # plotting
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(x, y, 'salmon', marker='o')

    # sort the data in ascending order
    x = np.sort(WT_con_B_cr_D1[idx])
    # get the cdf values of y
    y = np.arange(len(WT_con_B_cr_D1[idx])) / float(len(WT_con_B_cr_D1[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('Day 1')
    plt.plot(x, y, 'darkturquoise', marker='o')

    # CDF day 5 ----------------------------------------------
    stat_lev = stats.ks_2samp(WT_con_A_cr_D5[idx], WT_con_B_cr_D5[idx])

    # sort the data in ascending order
    x = np.sort(WT_con_A_cr_D5[idx])
    # get the cdf values of y
    y = np.arange(len(WT_con_A_cr_D5[idx])) / float(len(WT_con_A_cr_D5[idx]))

    # plotting
    plt.subplot(132)
    plt.plot(x, y, 'salmon', marker='o')

    # sort the data in ascending order
    x = np.sort(WT_con_B_cr_D5[idx])
    # get the cdf values of y
    y = np.arange(len(WT_con_B_cr_D5[idx])) / float(len(WT_con_B_cr_D5[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('Day 5')
    plt.plot(x, y, 'darkturquoise', marker='o')

    # CDF Day 9 ----------------------------------------------
    stat_lev = stats.ks_2samp(WT_con_A_cr_D9[idx], WT_con_B_cr_D9[idx])

    # sort the data in ascending order
    x = np.sort(WT_con_A_cr_D9[idx])
    # get the cdf values of y
    y = np.arange(len(WT_con_A_cr_D9[idx])) / float(len(WT_con_A_cr_D9[idx]))

    # plotting
    plt.subplot(133)
    plt.plot(x, y, 'salmon', marker='o')

    # sort the data in ascending order
    x = np.sort(WT_con_B_cr_D9[idx])
    # get the cdf values of y
    y = np.arange(len(WT_con_B_cr_D9[idx])) / float(len(WT_con_B_cr_D9[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('Day 9')
    plt.plot(x, y, 'darkturquoise', marker='o')

    plt.suptitle(f'{WT_mouse_id_indices[idx]} correlated Pairs Ratio CDF w/ sorted data [Pearson r val: {threshold}]')
    plt.savefig(os.path.join(os.getcwd(), f"visualization/20210526/{WT_mouse_id_indices[idx]}_corr_pair_CDF.png"), transparent=True, dpi=300)
    plt.show()

#%% Th Correlated Pairs Plots: con A v con B
for idx in range(len(Th_mouse_id_indices)):
    # CDF prestim, stim ----------------------------------------------
    stat_lev = stats.ks_2samp(Th_con_A_cr_D1[idx], Th_con_B_cr_D1[idx])

    # sort the data in ascending order
    x = np.sort(Th_con_A_cr_D1[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_A_cr_D1[idx])) / float(len(Th_con_A_cr_D1[idx]))

    # plotting
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(x, y, 'salmon', marker='o')

    # sort the data in ascending order
    x = np.sort(Th_con_B_cr_D1[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_B_cr_D1[idx])) / float(len(Th_con_B_cr_D1[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('Day 1')
    plt.plot(x, y, 'darkturquoise', marker='o')

    # CDF day 5 ----------------------------------------------
    stat_lev = stats.ks_2samp(Th_con_A_cr_D5[idx], Th_con_B_cr_D5[idx])

    # sort the data in ascending order
    x = np.sort(Th_con_A_cr_D5[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_A_cr_D5[idx])) / float(len(Th_con_A_cr_D5[idx]))

    # plotting
    plt.subplot(132)
    plt.plot(x, y, 'salmon', marker='o')

    # sort the data in ascending order
    x = np.sort(Th_con_B_cr_D5[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_B_cr_D5[idx])) / float(len(Th_con_B_cr_D5[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('Day 5')
    plt.plot(x, y, 'darkturquoise', marker='o')

    # CDF Day 9 ----------------------------------------------
    stat_lev = stats.ks_2samp(Th_con_A_cr_D9[idx], Th_con_B_cr_D9[idx])

    # sort the data in ascending order
    x = np.sort(Th_con_A_cr_D9[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_A_cr_D9[idx])) / float(len(Th_con_A_cr_D9[idx]))

    # plotting
    plt.subplot(133)
    plt.plot(x, y, 'salmon', marker='o')

    # sort the data in ascending order
    x = np.sort(Th_con_B_cr_D9[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_B_cr_D9[idx])) / float(len(Th_con_B_cr_D9[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('Day 9')
    plt.plot(x, y, 'darkturquoise', marker='o')

    plt.suptitle(f'{Th_mouse_id_indices[idx]} correlated Pairs Ratio CDF w/ sorted data [Pearson r val: {threshold}]')
    plt.savefig(os.path.join(os.getcwd(), f"visualization/20210526/TH_{Th_mouse_id_indices[idx]}_corr_pair_CDF_A_v_B.png"), transparent=True, dpi=300)
    plt.show()

#%% WT Correlated Pairs Plots: D1 v D9
for idx in range(len(WT_mouse_id_indices)):
    # CDF context A D1 v D9 ----------------------------------------------
    stat_lev = stats.ks_2samp(WT_con_A_cr_D1[idx], WT_con_A_cr_D9[idx])

    # sort the data in ascending order
    x = np.sort(WT_con_A_cr_D1[idx])
    # get the cdf values of y
    y = np.arange(len(WT_con_A_cr_D1[idx])) / float(len(WT_con_A_cr_D1[idx]))

    # plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(x, y, 'mistyrose', marker='o')

    # sort the data in ascending order
    x = np.sort(WT_con_A_cr_D9[idx])
    # get the cdf values of y
    y = np.arange(len(WT_con_A_cr_D9[idx])) / float(len(WT_con_A_cr_D9[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('context A')
    plt.plot(x, y, 'salmon', marker='o')

    # CDF context B D1 v D9 ----------------------------------------------
    stat_lev = stats.ks_2samp(WT_con_B_cr_D1[idx], WT_con_B_cr_D9[idx])

    # sort the data in ascending order
    x = np.sort(WT_con_B_cr_D1[idx])
    # get the cdf values of y
    y = np.arange(len(WT_con_B_cr_D1[idx])) / float(len(WT_con_B_cr_D1[idx]))

    # plotting
    plt.subplot(122)
    plt.plot(x, y, 'paleturquoise', marker='o')

    # sort the data in ascending order
    x = np.sort(WT_con_B_cr_D9[idx])
    # get the cdf values of y
    y = np.arange(len(WT_con_B_cr_D9[idx])) / float(len(WT_con_B_cr_D9[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('context B')
    plt.plot(x, y, 'darkturquoise', marker='o')

    plt.suptitle(f'{WT_mouse_id_indices[idx]} correlated Pairs Ratio CDF w/ sorted data [Pearson r val: {threshold}]')
    plt.savefig(os.path.join(os.getcwd(), f"visualization/20210526/WT_{WT_mouse_id_indices[idx]}_corr_pair_CDF_D1_v_D9.png"), transparent=True, dpi=300)
    plt.show()

#%% Th Correlated Pairs Plots: D1 v D9
for idx in range(len(Th_mouse_id_indices)):
    # CDF context A D1 v D9 ----------------------------------------------
    stat_lev = stats.ks_2samp(Th_con_A_cr_D1[idx], Th_con_A_cr_D9[idx])

    # sort the data in ascending order
    x = np.sort(Th_con_A_cr_D1[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_A_cr_D1[idx])) / float(len(Th_con_A_cr_D1[idx]))

    # plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(x, y, 'mistyrose', marker='o')

    # sort the data in ascending order
    x = np.sort(Th_con_A_cr_D9[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_A_cr_D9[idx])) / float(len(Th_con_A_cr_D9[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('context A')
    plt.plot(x, y, 'salmon', marker='o')

    # CDF context B D1 v D9 ----------------------------------------------
    stat_lev = stats.ks_2samp(Th_con_B_cr_D1[idx], Th_con_B_cr_D9[idx])

    # sort the data in ascending order
    x = np.sort(Th_con_B_cr_D1[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_B_cr_D1[idx])) / float(len(Th_con_B_cr_D1[idx]))

    # plotting
    plt.subplot(122)
    plt.plot(x, y, 'paleturquoise', marker='o')

    # sort the data in ascending order
    x = np.sort(Th_con_B_cr_D9[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_B_cr_D9[idx])) / float(len(Th_con_B_cr_D9[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('context B')
    plt.plot(x, y, 'darkturquoise', marker='o')

    plt.suptitle(f'{Th_mouse_id_indices[idx]} correlated Pairs Ratio CDF w/ sorted data [Pearson r val: {threshold}]')
    plt.savefig(os.path.join(os.getcwd(), f"visualization/20210526/Th_{Th_mouse_id_indices[idx]}_corr_pair_CDF_D1_v_D9.png"), transparent=True, dpi=300)
    plt.show()

#%% WT Clustering Coefficient Plots: D1 v D9
for idx in range(len(WT_mouse_id_indices)):
    # CDF context A D1 v D9 ----------------------------------------------
    stat_lev = stats.ks_2samp(WT_con_A_cc_D1[idx], WT_con_A_cc_D9[idx])

    # sort the data in ascending order
    x = np.sort(WT_con_A_cc_D1[idx])
    # get the cdf values of y
    y = np.arange(len(WT_con_A_cc_D1[idx])) / float(len(WT_con_A_cc_D1[idx]))

    # plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(x, y, 'mistyrose', marker='o')

    # sort the data in ascending order
    x = np.sort(WT_con_A_cc_D9[idx])
    # get the cdf values of y
    y = np.arange(len(WT_con_A_cc_D9[idx])) / float(len(WT_con_A_cc_D9[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('context A')
    plt.plot(x, y, 'salmon', marker='o')

    # CDF context B D1 v D9 ----------------------------------------------
    stat_lev = stats.ks_2samp(WT_con_B_cc_D1[idx], WT_con_B_cc_D9[idx])

    # sort the data in ascending order
    x = np.sort(WT_con_B_cc_D1[idx])
    # get the cdf values of y
    y = np.arange(len(WT_con_B_cc_D1[idx])) / float(len(WT_con_B_cc_D1[idx]))

    # plotting
    plt.subplot(122)
    plt.plot(x, y, 'paleturquoise', marker='o')

    # sort the data in ascending order
    x = np.sort(WT_con_B_cc_D9[idx])
    # get the cdf values of y
    y = np.arange(len(WT_con_B_cc_D9[idx])) / float(len(WT_con_B_cc_D9[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('context B')
    plt.plot(x, y, 'darkturquoise', marker='o')

    plt.suptitle(f'{WT_mouse_id_indices[idx]} Clustering Coefficient Ratio CDF w/ sorted data [Pearson r val: {threshold}]')
    plt.savefig(os.path.join(os.getcwd(), f"visualization/20210526/WT_{WT_mouse_id_indices[idx]}_clustering_coefficient_CDF_D1_v_D9.png"), transparent=True, dpi=300)
    plt.show()

#%% Th Clustering Coefficient Plots: D1 v D9
for idx in range(len(Th_mouse_id_indices)):
    # CDF context A D1 v D9 ----------------------------------------------
    stat_lev = stats.ks_2samp(Th_con_A_cc_D1[idx], Th_con_A_cc_D9[idx])

    # sort the data in ascending order
    x = np.sort(Th_con_A_cc_D1[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_A_cc_D1[idx])) / float(len(Th_con_A_cc_D1[idx]))

    # plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(x, y, 'mistyrose', marker='o')

    # sort the data in ascending order
    x = np.sort(Th_con_A_cc_D9[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_A_cc_D9[idx])) / float(len(Th_con_A_cc_D9[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('context A')
    plt.plot(x, y, 'salmon', marker='o')

    # CDF context B D1 v D9 ----------------------------------------------
    stat_lev = stats.ks_2samp(Th_con_B_cc_D1[idx], Th_con_B_cc_D9[idx])

    # sort the data in ascending order
    x = np.sort(Th_con_B_cc_D1[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_B_cc_D1[idx])) / float(len(Th_con_B_cc_D1[idx]))

    # plotting
    plt.subplot(122)
    plt.plot(x, y, 'paleturquoise', marker='o')

    # sort the data in ascending order
    x = np.sort(Th_con_B_cc_D9[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_B_cc_D9[idx])) / float(len(Th_con_B_cc_D9[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('context B')
    plt.plot(x, y, 'darkturquoise', marker='o')

    plt.suptitle(f'{Th_mouse_id_indices[idx]} Clustering Coefficient Ratio CDF w/ sorted data [Pearson r val: {threshold}]')
    plt.savefig(os.path.join(os.getcwd(), f"visualization/20210526/Th_{Th_mouse_id_indices[idx]}_clustering_coefficient_CDF_D1_v_D9.png"), transparent=True, dpi=300)
    plt.show()



#%% WT Clustering Coefficient Plots: D0 v D9
for idx in range(len(WT_mouse_id_indices)):
    # CDF context A D1 v D9 ----------------------------------------------
    stat_lev = stats.ks_2samp(WT_con_A_cc_D0[idx], WT_con_A_cc_D9[idx])

    # sort WTe data in ascending order
    x = np.sort(WT_con_A_cc_D0[idx])
    # get WTe cdf values of y
    y = np.arange(len(WT_con_A_cc_D0[idx])) / float(len(WT_con_A_cc_D0[idx]))

    # plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(x, y, 'lightgrey', marker='o')

    # sort WTe data in ascending order
    x = np.sort(WT_con_A_cc_D9[idx])
    # get WTe cdf values of y
    y = np.arange(len(WT_con_A_cc_D9[idx])) / float(len(WT_con_A_cc_D9[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('context A')
    plt.plot(x, y, 'salmon', marker='o')

    # CDF context B D1 v D9 ----------------------------------------------
    stat_lev = stats.ks_2samp(WT_con_B_cc_D0[idx], WT_con_B_cc_D9[idx])

    # sort WTe data in ascending order
    x = np.sort(WT_con_B_cc_D0[idx])
    # get WTe cdf values of y
    y = np.arange(len(WT_con_B_cc_D0[idx])) / float(len(WT_con_B_cc_D0[idx]))

    # plotting
    plt.subplot(122)
    plt.plot(x, y, 'lightgrey', marker='o')

    # sort WTe data in ascending order
    x = np.sort(WT_con_B_cc_D9[idx])
    # get WTe cdf values of y
    y = np.arange(len(WT_con_B_cc_D9[idx])) / float(len(WT_con_B_cc_D9[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('context B')
    plt.plot(x, y, 'darkturquoise', marker='o')

    plt.suptitle(f'{WT_mouse_id_indices[idx]} Clustering Coefficient Ratio CDF w/ sorted data [Pearson r val: {threshold}]')
    plt.savefig(os.path.join(os.getcwd(), f"visualization/20210526/WT_{WT_mouse_id_indices[idx]}_clustering_coefficient_CDF_D0_v_D9.png"), transparent=True, dpi=300)
    plt.show()

#%% Th Clustering Coefficient Plots: D0 v D9
for idx in range(len(Th_mouse_id_indices)):
    # CDF context A D1 v D9 ----------------------------------------------
    stat_lev = stats.ks_2samp(Th_con_A_cc_D0[idx], Th_con_A_cc_D9[idx])

    # sort the data in ascending order
    x = np.sort(Th_con_A_cc_D0[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_A_cc_D0[idx])) / float(len(Th_con_A_cc_D0[idx]))

    # plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.plot(x, y, 'lightgrey', marker='o')

    # sort the data in ascending order
    x = np.sort(Th_con_A_cc_D9[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_A_cc_D9[idx])) / float(len(Th_con_A_cc_D9[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('context A')
    plt.plot(x, y, 'salmon', marker='o')

    # CDF context B D1 v D9 ----------------------------------------------
    stat_lev = stats.ks_2samp(Th_con_B_cc_D0[idx], Th_con_B_cc_D9[idx])

    # sort the data in ascending order
    x = np.sort(Th_con_B_cc_D0[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_B_cc_D0[idx])) / float(len(Th_con_B_cc_D0[idx]))

    # plotting
    plt.subplot(122)
    plt.plot(x, y, 'lightgrey', marker='o')

    # sort the data in ascending order
    x = np.sort(Th_con_B_cc_D9[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_B_cc_D9[idx])) / float(len(Th_con_B_cc_D9[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('context B')
    plt.plot(x, y, 'darkturquoise', marker='o')

    plt.suptitle(f'{Th_mouse_id_indices[idx]} Clustering Coefficient Ratio CDF w/ sorted data [Pearson r val: {threshold}]')
    plt.savefig(os.path.join(os.getcwd(), f"visualization/20210526/Th_{Th_mouse_id_indices[idx]}_clustering_coefficient_CDF_D0_v_D9.png"), transparent=True, dpi=300)
    plt.show()

#%% WT Clustering Coefficient con A v con B
for idx in range(len(WT_mouse_id_indices)):
    # CDF prestim, stim ----------------------------------------------
    stat_lev = stats.ks_2samp(WT_con_A_cc_D1[idx], WT_con_B_cc_D1[idx])

    # sort the data in ascending order
    x = np.sort(WT_con_A_cc_D1[idx])
    # get the cdf values of y
    y = np.arange(len(WT_con_A_cc_D1[idx])) / float(len(WT_con_A_cc_D1[idx]))

    # plotting
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(x, y, 'salmon', marker='o')

    # sort the data in ascending order
    x = np.sort(WT_con_B_cc_D1[idx])
    # get the cdf values of y
    y = np.arange(len(WT_con_B_cc_D1[idx])) / float(len(WT_con_B_cc_D1[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('Day 1')
    plt.plot(x, y, 'darkturquoise', marker='o')

    # CDF day 5 ----------------------------------------------
    stat_lev = stats.ks_2samp(WT_con_A_cc_D5[idx], WT_con_B_cc_D5[idx])

    # sort the data in ascending order
    x = np.sort(WT_con_A_cc_D5[idx])
    # get the cdf values of y
    y = np.arange(len(WT_con_A_cc_D5[idx])) / float(len(WT_con_A_cc_D5[idx]))

    # plotting
    plt.subplot(132)
    plt.plot(x, y, 'salmon', marker='o')

    # sort the data in ascending order
    x = np.sort(WT_con_B_cc_D5[idx])
    # get the cdf values of y
    y = np.arange(len(WT_con_B_cc_D5[idx])) / float(len(WT_con_B_cc_D5[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('Day 5')
    plt.plot(x, y, 'darkturquoise', marker='o')

    # CDF Day 9 ----------------------------------------------
    stat_lev = stats.ks_2samp(WT_con_A_cc_D9[idx], WT_con_B_cc_D9[idx])

    # sort the data in ascending order
    x = np.sort(WT_con_A_cc_D9[idx])
    # get the cdf values of y
    y = np.arange(len(WT_con_A_cc_D9[idx])) / float(len(WT_con_A_cc_D9[idx]))

    # plotting
    plt.subplot(133)
    plt.plot(x, y, 'salmon', marker='o')

    # sort the data in ascending order
    x = np.sort(WT_con_B_cc_D9[idx])
    # get the cdf values of y
    y = np.arange(len(WT_con_B_cc_D9[idx])) / float(len(WT_con_B_cc_D9[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('Day 9')
    plt.plot(x, y, 'darkturquoise', marker='o')

    plt.suptitle(f'{WT_mouse_id_indices[idx]} clustering coefficient CDF w/ sorted data [Pearson r val: {threshold}]')
    plt.savefig(os.path.join(os.getcwd(), f"visualization/20210526/WT_{WT_mouse_id_indices[idx]}_clustering_coefficient_CDF_A_v_B.png"), transparent=True, dpi=300)
    plt.show()

#%% Th Clustering Coefficient con A v con B
for idx in range(len(Th_mouse_id_indices)):
    # CDF prestim, stim ----------------------------------------------
    stat_lev = stats.ks_2samp(Th_con_A_cc_D1[idx], Th_con_B_cc_D1[idx])

    # sort the data in ascending order
    x = np.sort(Th_con_A_cc_D1[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_A_cc_D1[idx])) / float(len(Th_con_A_cc_D1[idx]))

    # plotting
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(x, y, 'salmon', marker='o')

    # sort the data in ascending order
    x = np.sort(Th_con_B_cc_D1[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_B_cc_D1[idx])) / float(len(Th_con_B_cc_D1[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('Day 1')
    plt.plot(x, y, 'darkturquoise', marker='o')

    # CDF day 5 ----------------------------------------------
    stat_lev = stats.ks_2samp(Th_con_A_cc_D5[idx], Th_con_B_cc_D5[idx])

    # sort the data in ascending order
    x = np.sort(Th_con_A_cc_D5[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_A_cc_D5[idx])) / float(len(Th_con_A_cc_D5[idx]))

    # plotting
    plt.subplot(132)
    plt.plot(x, y, 'salmon', marker='o')

    # sort the data in ascending order
    x = np.sort(Th_con_B_cc_D5[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_B_cc_D5[idx])) / float(len(Th_con_B_cc_D5[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('Day 5')
    plt.plot(x, y, 'darkturquoise', marker='o')

    # CDF Day 9 ----------------------------------------------
    stat_lev = stats.ks_2samp(Th_con_A_cc_D9[idx], Th_con_B_cc_D9[idx])

    # sort the data in ascending order
    x = np.sort(Th_con_A_cc_D9[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_A_cc_D9[idx])) / float(len(Th_con_A_cc_D9[idx]))

    # plotting
    plt.subplot(133)
    plt.plot(x, y, 'salmon', marker='o')

    # sort the data in ascending order
    x = np.sort(Th_con_B_cc_D9[idx])
    # get the cdf values of y
    y = np.arange(len(Th_con_B_cc_D9[idx])) / float(len(Th_con_B_cc_D9[idx]))

    # plotting
    plt.xlabel(f'P value: {stat_lev.pvalue:.2e}')
    plt.ylabel('CDF')

    plt.title('Day 9')
    plt.plot(x, y, 'darkturquoise', marker='o')

    plt.suptitle(f'{Th_mouse_id_indices[idx]} clustering coefficient CDF w/ sorted data [Pearson r val: {threshold}]')
    plt.savefig(os.path.join(os.getcwd(), f"visualization/20210526/Th_{Th_mouse_id_indices[idx]}_clustering_coefficient_CDF_A_v_B.png"), transparent=True, dpi=300)
    plt.show()



#%% Plot all mice trajectories to examine differences
#Th Clustering Coefficient
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#1727AE", "#2C43B8", "#445BC1", "#6987D5", "#97BAEC", "#BBDFFA"])
fig = plt.figure(figsize=(15,10))
for idx in range(len(Th_mouse_id_indices)):
    plt.subplot(221)
    # sort WTe data in ascending order
    x = np.sort(Th_con_A_cc_D1[idx])
    # get WTe cdf values of y
    y = np.arange(len(Th_con_A_cc_D1[idx])) / float(len(Th_con_A_cc_D1[idx]))

    # plotting
    plt.ylabel('CDF')

    plt.title('context A, day 1')
    plt.plot(x, y, marker='o')

    plt.subplot(223)
    # sort WTe data in ascending order
    x = np.sort(Th_con_A_cc_D9[idx])
    # get WTe cdf values of y
    y = np.arange(len(Th_con_A_cc_D9[idx])) / float(len(Th_con_A_cc_D9[idx]))

    # plotting
    plt.ylabel('CDF')

    plt.title('context A, day 9')
    plt.plot(x, y, marker='o')

    plt.subplot(222)
    # sort WTe data in ascending order
    x = np.sort(Th_con_B_cc_D1[idx])
    # get WTe cdf values of y
    y = np.arange(len(Th_con_B_cc_D1[idx])) / float(len(Th_con_B_cc_D1[idx]))

    # plotting
    plt.ylabel('CDF')
    plt.title('context B, day 1')
    plt.plot(x, y,marker='o')

    plt.subplot(224)
    # sort WTe data in ascending order
    x = np.sort(Th_con_B_cc_D9[idx])
    # get WTe cdf values of y
    y = np.arange(len(Th_con_B_cc_D9[idx])) / float(len(Th_con_B_cc_D9[idx]))

    # plotting
    plt.ylabel('CDF')

    plt.title('context B, day 9')
    plt.plot(x, y, marker='o')

plt.figlegend(labels=Th_mouse_id_indices)
plt.suptitle(f'Th Clustering Coefficient Ratio CDF w/ sorted data [Pearson r val: {threshold}]')
plt.savefig(os.path.join(os.getcwd(), f"visualization/20210526/Th_clustering_coefficient_CDF_all_mice.png"), transparent=True, dpi=300)
plt.show()


#%% WT Clustering Coefficient
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#1727AE", "#2C43B8", "#445BC1", "#6987D5", "#97BAEC", "#BBDFFA"])
fig = plt.figure(figsize=(15,10))
for idx in range(len(WT_mouse_id_indices)):
    plt.subplot(221)
    # sort WTe data in ascending order
    x = np.sort(WT_con_A_cc_D1[idx])
    # get WTe cdf values of y
    y = np.arange(len(WT_con_A_cc_D1[idx])) / float(len(WT_con_A_cc_D1[idx]))

    # plotting
    plt.ylabel('CDF')

    plt.title('context A, day 1')
    plt.plot(x, y, marker='o')

    plt.subplot(223)
    # sort WTe data in ascending order
    x = np.sort(WT_con_A_cc_D9[idx])
    # get WTe cdf values of y
    y = np.arange(len(WT_con_A_cc_D9[idx])) / float(len(WT_con_A_cc_D9[idx]))

    # plotting
    plt.ylabel('CDF')

    plt.title('context A, day 9')
    plt.plot(x, y, marker='o')

    plt.subplot(222)
    # sort WTe data in ascending order
    x = np.sort(WT_con_B_cc_D1[idx])
    # get WTe cdf values of y
    y = np.arange(len(WT_con_B_cc_D1[idx])) / float(len(WT_con_B_cc_D1[idx]))

    # plotting
    plt.ylabel('CDF')
    plt.title('context B, day 1')
    plt.plot(x, y,marker='o')

    plt.subplot(224)
    # sort WTe data in ascending order
    x = np.sort(WT_con_B_cc_D9[idx])
    # get WTe cdf values of y
    y = np.arange(len(WT_con_B_cc_D9[idx])) / float(len(WT_con_B_cc_D9[idx]))

    # plotting
    plt.ylabel('CDF')

    plt.title('context B, day 9')
    plt.plot(x, y, marker='o')

plt.figlegend(labels=WT_mouse_id_indices)
plt.suptitle(f'WT Clustering Coefficient Ratio CDF w/ sorted data [Pearson r val: {threshold}]')
plt.savefig(os.path.join(os.getcwd(), f"visualization/20210526/WT_clustering_coefficient_CDF_all_mice.png"), transparent=True, dpi=300)
plt.show()
