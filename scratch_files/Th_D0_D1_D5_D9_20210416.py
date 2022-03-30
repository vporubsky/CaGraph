"""
Created on April 15, 2021

@author: Veronica Porubsky

Title: Run batch analyses, collecting distributions

Context A - anxiogenic
Context B - neutral
"""
from neuronal_network_graph import DGNetworkGraph as nng
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

sns.set(style="whitegrid")

# %% Load treated data files - Th
# D0_Th = ['2-1_D0_smoothed_calcium_traces.csv', '2-2_D0_smoothed_calcium_traces.csv',
#          '2-3_D0_smoothed_calcium_traces.csv', '348-1_D0_smoothed_calcium_traces.csv',
#          '349-2_D0_smoothed_calcium_traces.csv', '386-2_D0_smoothed_calcium_traces.csv',
#          '387-4_D0_smoothed_calcium_traces.csv', '396-1_D0_smoothed_calcium_traces.csv',
#          '396-3_D0_smoothed_calcium_traces.csv']
D0_Th = ['348-1_D0_smoothed_calcium_traces.csv',
         '349-2_D0_smoothed_calcium_traces.csv', '386-2_D0_smoothed_calcium_traces.csv',
         '387-4_D0_smoothed_calcium_traces.csv', '396-1_D0_smoothed_calcium_traces.csv',
         '396-3_D0_smoothed_calcium_traces.csv']
# D1_Th = ['2-1_D1_smoothed_calcium_traces.csv', '2-2_D1_smoothed_calcium_traces.csv',
#          '2-3_D1_smoothed_calcium_traces.csv', '348-1_D1_smoothed_calcium_traces.csv',
#          '349-2_D1_smoothed_calcium_traces.csv', '386-2_D1_smoothed_calcium_traces.csv',
#          '387-4_D1_smoothed_calcium_traces.csv', '396-1_D1_smoothed_calcium_traces.csv',
#          '396-3_D1_smoothed_calcium_traces.csv']
D1_Th = ['348-1_D1_smoothed_calcium_traces.csv',
         '349-2_D1_smoothed_calcium_traces.csv', '386-2_D1_smoothed_calcium_traces.csv',
         '387-4_D1_smoothed_calcium_traces.csv', '396-1_D1_smoothed_calcium_traces.csv',
         '396-3_D1_smoothed_calcium_traces.csv']
# D5_Th = ['2-1_D5_smoothed_calcium_traces.csv', '2-2_D5_smoothed_calcium_traces.csv',
#          '2-3_D5_smoothed_calcium_traces.csv', '348-1_D5_smoothed_calcium_traces.csv',
#          '349-2_D5_smoothed_calcium_traces.csv', '386-2_D5_smoothed_calcium_traces.csv',
#          '387-4_D5_smoothed_calcium_traces.csv', '396-1_D5_smoothed_calcium_traces.csv',
#          '396-3_D5_smoothed_calcium_traces.csv']
D5_Th = ['348-1_D5_smoothed_calcium_traces.csv',
         '349-2_D5_smoothed_calcium_traces.csv', '386-2_D5_smoothed_calcium_traces.csv',
         '387-4_D5_smoothed_calcium_traces.csv', '396-1_D5_smoothed_calcium_traces.csv',
         '396-3_D5_smoothed_calcium_traces.csv']
# D9_Th = ['2-1_D9_smoothed_calcium_traces.csv', '2-2_D9_smoothed_calcium_traces.csv',
#          '2-3_D9_smoothed_calcium_traces.csv', '348-1_D9_smoothed_calcium_traces.csv',
#          '349-2_D9_smoothed_calcium_traces.csv', '386-2_D9_smoothed_calcium_traces.csv',
#          '387-4_D9_smoothed_calcium_traces.csv', '396-1_D9_smoothed_calcium_traces.csv',
#          '396-3_D9_smoothed_calcium_traces.csv']
D9_Th = ['348-1_D9_smoothed_calcium_traces.csv',
         '349-2_D9_smoothed_calcium_traces.csv', '386-2_D9_smoothed_calcium_traces.csv',
         '387-4_D9_smoothed_calcium_traces.csv', '396-1_D9_smoothed_calcium_traces.csv',
         '396-3_D9_smoothed_calcium_traces.csv']

all_Th_files = [D0_Th, D1_Th, D5_Th, D9_Th]

# %% All measurements, separating contexts
threshold = 0.3
names = []
data_mat = []

con_A_subnetworks_D0 = []
con_B_subnetworks_D0 = []
con_A_subnetworks_D1 = []
con_B_subnetworks_D1 = []
con_A_subnetworks_D5 = []
con_B_subnetworks_D5 = []
con_A_subnetworks_D9 = []
con_B_subnetworks_D9 = []

con_A_num_subnetworks_D0 = []
con_B_num_subnetworks_D0 = []
con_A_num_subnetworks_D1 = []
con_B_num_subnetworks_D1 = []
con_A_num_subnetworks_D5 = []
con_B_num_subnetworks_D5 = []
con_A_num_subnetworks_D9 = []
con_B_num_subnetworks_D9 = []

con_A_size_subnetworks_D0 = []
con_B_size_subnetworks_D0 = []
con_A_size_subnetworks_D1 = []
con_B_size_subnetworks_D1 = []
con_A_size_subnetworks_D5 = []
con_B_size_subnetworks_D5 = []
con_A_size_subnetworks_D9 = []
con_B_size_subnetworks_D9 = []

con_A_cc_D0 = []
con_B_cc_D0 = []
con_A_cc_D1 = []
con_B_cc_D1 = []
con_A_cc_D5 = []
con_B_cc_D5 = []
con_A_cc_D9 = []
con_B_cc_D9 = []

con_A_hubs_D0 = []
con_B_hubs_D0 = []
con_A_hubs_D1 = []
con_B_hubs_D1 = []
con_A_hubs_D5 = []
con_B_hubs_D5 = []
con_A_hubs_D9 = []
con_B_hubs_D9 = []

con_A_num_hubs_D0 = []
con_B_num_hubs_D0 = []
con_A_num_hubs_D1 = []
con_B_num_hubs_D1 = []
con_A_num_hubs_D5 = []
con_B_num_hubs_D5 = []
con_A_num_hubs_D9 = []
con_B_num_hubs_D9 = []

con_A_hits_D0 = []
con_B_hits_D0 = []
con_A_hits_D1 = []
con_B_hits_D1 = []
con_A_hits_D5 = []
con_B_hits_D5 = []
con_A_hits_D9 = []
con_B_hits_D9 = []

con_A_cr_D0 = []
con_B_cr_D0 = []
con_A_cr_D1 = []
con_B_cr_D1 = []
con_A_cr_D5 = []
con_B_cr_D5 = []
con_A_cr_D9 = []
con_B_cr_D9 = []

edges_A_D0 = []
edges_B_D0 = []
edges_A_D1 = []
edges_B_D1 = []
edges_A_D5 = []
edges_B_D5 = []
edges_A_D9 = []
edges_B_D9 = []

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

        # correlated pairs ratio
        cr_A = nn.get_context_A_correlated_pair_ratio(threshold=threshold)
        cr_B = nn.get_context_B_correlated_pair_ratio(threshold=threshold)

        # communities
        # community_A = list(nx.algorithms.community.greedy_modularity_communities(conA))
        # community_B = list(nx.algorithms.community.greedy_modularity_communities(conB))

        if day == 0:
            con_A_subnetworks_D0.append(connected_subnetworks_A)
            con_B_subnetworks_D0.append(connected_subnetworks_B)
            con_A_num_subnetworks_D0.append(num_connected_subnetworks_A / num_neurons)
            con_B_num_subnetworks_D0.append(num_connected_subnetworks_B / num_neurons)
            con_A_size_subnetworks_D0.append(np.median(len_connected_subnetworks_A) / num_neurons)
            con_B_size_subnetworks_D0.append(np.median(len_connected_subnetworks_B) / num_neurons)
            edges_A_D0.append(list(conA.edges()))
            edges_B_D0.append(list(conB.edges()))
            con_A_hubs_D0.append(hubs_A)
            con_B_hubs_D0.append(hubs_B)
            con_A_num_hubs_D0.append(len_hubs_A / num_neurons)
            con_B_num_hubs_D0.append(len_hubs_B / num_neurons)
            con_A_cc_D0.append(cc_A)
            con_B_cc_D0.append(cc_B)
            con_A_hits_D0.append([hit_val/num_neurons for hit_val in list(hits_A.values())])
            con_B_hits_D0.append([hit_val/num_neurons for hit_val in list(hits_B.values())])
            con_A_cr_D0.append(cr_A)
            con_B_cr_D0.append(cr_B)

        elif day == 1:
            con_A_subnetworks_D1.append(connected_subnetworks_A)
            con_B_subnetworks_D1.append(connected_subnetworks_B)
            con_A_num_subnetworks_D1.append(num_connected_subnetworks_A / num_neurons)
            con_B_num_subnetworks_D1.append(num_connected_subnetworks_B / num_neurons)
            con_A_size_subnetworks_D1.append(np.median(len_connected_subnetworks_A) / num_neurons)
            con_B_size_subnetworks_D1.append(np.median(len_connected_subnetworks_B) / num_neurons)
            edges_A_D1.append(list(conA.edges()))
            edges_B_D1.append(list(conB.edges()))
            con_A_hubs_D1.append(hubs_A)
            con_B_hubs_D1.append(hubs_B)
            con_A_num_hubs_D1.append(len_hubs_A / num_neurons)
            con_B_num_hubs_D1.append(len_hubs_B / num_neurons)
            con_A_cc_D1.append(cc_A)
            con_B_cc_D1.append(cc_B)
            con_A_hits_D1.append([hit_val/num_neurons for hit_val in list(hits_A.values())])
            con_B_hits_D1.append([hit_val/num_neurons for hit_val in list(hits_B.values())])
            con_A_cr_D1.append(cr_A)
            con_B_cr_D1.append(cr_B)

        elif day == 2:
            con_A_subnetworks_D5.append(connected_subnetworks_A)
            con_B_subnetworks_D5.append(connected_subnetworks_B)
            con_A_num_subnetworks_D5.append(num_connected_subnetworks_A / num_neurons)
            con_B_num_subnetworks_D5.append(num_connected_subnetworks_B / num_neurons)
            con_A_size_subnetworks_D5.append(np.median(len_connected_subnetworks_A) / num_neurons)
            con_B_size_subnetworks_D5.append(np.median(len_connected_subnetworks_B) / num_neurons)
            edges_A_D5.append(list(conA.edges()))
            edges_B_D5.append(list(conB.edges()))
            con_A_hubs_D5.append(hubs_A)
            con_B_hubs_D5.append(hubs_B)
            con_A_num_hubs_D5.append(len_hubs_A / num_neurons)
            con_B_num_hubs_D5.append(len_hubs_B / num_neurons)
            con_A_cc_D5.append(cc_A)
            con_B_cc_D5.append(cc_B)
            con_A_hits_D5.append(list(hits_A.values()))
            con_B_hits_D5.append(list(hits_B.values()))
            con_A_cr_D5.append(cr_A)
            con_B_cr_D5.append(cr_B)

        elif day == 3:
            con_A_subnetworks_D9.append(connected_subnetworks_A)
            con_B_subnetworks_D9.append(connected_subnetworks_B)
            con_A_num_subnetworks_D9.append(num_connected_subnetworks_A / num_neurons)
            con_B_num_subnetworks_D9.append(num_connected_subnetworks_B / num_neurons)
            con_A_size_subnetworks_D9.append(np.median(len_connected_subnetworks_A) / num_neurons)
            con_B_size_subnetworks_D9.append(np.median(len_connected_subnetworks_B) / num_neurons)
            edges_A_D9.append(list(conA.edges()))
            edges_B_D9.append(list(conB.edges()))
            con_A_hubs_D9.append(hubs_A)
            con_B_hubs_D9.append(hubs_B)
            con_A_num_hubs_D9.append(len_hubs_A / num_neurons)
            con_B_num_hubs_D9.append(len_hubs_B / num_neurons)
            con_A_cc_D9.append(cc_A)
            con_B_cc_D9.append(cc_B)
            con_A_hits_D9.append(list(hits_A.values()))
            con_B_hits_D9.append(list(hits_B.values()))
            con_A_cr_D9.append(cr_A)
            con_B_cr_D9.append(cr_B)

#%% HITS analysis
HITS_D0_A = [item for sublist in con_A_hits_D0 for item in sublist]
HITS_D0_B = [item for sublist in con_B_hits_D0 for item in sublist]

HITS_D1_A = [item for sublist in con_A_hits_D1 for item in sublist]
HITS_D1_B = [item for sublist in con_B_hits_D1 for item in sublist]

HITS_D5_A = [item for sublist in con_A_hits_D5 for item in sublist]
HITS_D5_B = [item for sublist in con_B_hits_D5 for item in sublist]

HITS_D9_A = [item for sublist in con_A_hits_D9 for item in sublist]
HITS_D9_B = [item for sublist in con_B_hits_D9 for item in sublist]

labels = ['D0_A', 'D0_B', 'D1_A', 'D1_B', 'D5_A', 'D5_B', 'D9_A', 'D9_B']
raw = [HITS_D0_A, HITS_D0_B, HITS_D1_A, HITS_D1_B, HITS_D5_A, HITS_D5_B, HITS_D9_A, HITS_D9_B]

plt.figure(figsize=(10, 15))
plt.subplot(421); plt.hist(raw[0], bins=20, color='grey', alpha=0.4); plt.title('Th_D0'); #plt.xlim((0,0.5))
# plt.subplot(422); plt.hist(raw[1], bins=20, color='b', alpha=0.4); plt.title('Th_D0_B'); plt.xlim((0,0.5))


plt.subplot(423); plt.hist(raw[2], bins=20, color='salmon', alpha=0.4); plt.title('Th_D1_A');# plt.xlim((0,0.5))
plt.subplot(424); plt.hist(raw[3], bins=20, color='turquoise', alpha=0.4); plt.title('Th_D1_B'); #plt.xlim((0,0.5))

plt.subplot(425); plt.hist(raw[4], bins=20, color='salmon', alpha=0.4); plt.title('Th_D5_A'); #plt.xlim((0,0.5))
plt.subplot(426); plt.hist(raw[5], bins=20, color='turquoise', alpha=0.4); plt.title('Th_D5_B');# plt.xlim((0,0.5))

plt.subplot(427); plt.hist(raw[6], bins=20, color='salmon', alpha=0.4); plt.title('Th_D9_A'); #plt.xlim((0,0.5))
plt.subplot(428); plt.hist(raw[7], bins=20, color='turquoise', alpha=0.4); plt.title('Th_D9_B'); #plt.xlim((0,0.5))

plt.suptitle(f'Hub value, Pearson r val: {threshold}')
plt.savefig(os.path.join(os.getcwd(), f"visualization/20210416/Th_hubs.png"), dpi=300)
plt.show()


#%% Correlated pairs ratio analysis
con_A_cr_D0 = [item for sublist in con_A_cr_D0 for item in sublist]
con_B_cr_D0 = [item for sublist in con_B_cr_D0 for item in sublist]

con_A_cr_D1 = [item for sublist in con_A_cr_D1 for item in sublist]
con_B_cr_D1 = [item for sublist in con_B_cr_D1 for item in sublist]

con_A_cr_D5 = [item for sublist in con_A_cr_D5 for item in sublist]
con_B_cr_D5 = [item for sublist in con_B_cr_D5 for item in sublist]

con_A_cr_D9 = [item for sublist in con_A_cr_D9 for item in sublist]
con_B_cr_D9 = [item for sublist in con_B_cr_D9 for item in sublist]

labels = ['D0_A', 'D0_B', 'D1_A', 'D1_B', 'D5_A', 'D5_B', 'D9_A', 'D9_B']
raw = [con_A_cr_D0, con_B_cr_D0, con_A_cr_D1, con_B_cr_D1, con_A_cr_D5, con_B_cr_D5, con_A_cr_D9, con_B_cr_D9]

plt.figure(figsize=(10, 15))
plt.subplot(421); plt.hist(raw[0], bins=20, color='grey', alpha=0.4); plt.title('Th_D0'); plt.xlim((0,0.5))
#plt.subplot(422); plt.hist(raw[1], bins=20, color='b', alpha=0.4); plt.title('Th_D0_B'); plt.xlim((0,0.5))


plt.subplot(423); plt.hist(raw[2], bins=50, color='salmon', alpha=0.4); plt.title('Th_D1_A'); plt.xlim((0,0.5))
plt.subplot(424); plt.hist(raw[3], bins=50, color='turquoise', alpha=0.4); plt.title('Th_D1_B'); plt.xlim((0,0.5))

plt.subplot(425); plt.hist(raw[4], bins=50, color='salmon', alpha=0.4); plt.title('Th_D5_A'); plt.xlim((0,0.5))
plt.subplot(426); plt.hist(raw[5], bins=50, color='turquoise', alpha=0.4); plt.title('Th_D5_B'); plt.xlim((0,0.5))

plt.subplot(427); plt.hist(raw[6], bins=50, color='salmon', alpha=0.4); plt.title('Th_D9_A'); plt.xlim((0,0.5))
plt.subplot(428); plt.hist(raw[7], bins=50, color='turquoise', alpha=0.4); plt.title('Th_D9_B'); plt.xlim((0,0.5))

plt.suptitle(f'Correlated pairs ratio, Pearson r val: {threshold}')
plt.savefig(os.path.join(os.getcwd(), f"visualization/20210416/Th_correlated_pairs.png"), dpi=300)
plt.show()

# %% clustering coefficient
con_A_cc_D0 = [item for sublist in con_A_cc_D0 for item in sublist]
con_B_cc_D0 = [item for sublist in con_B_cc_D0 for item in sublist]

con_A_cc_D1 = [item for sublist in con_A_cc_D1 for item in sublist]
con_B_cc_D1 = [item for sublist in con_B_cc_D1 for item in sublist]

con_A_cc_D5 = [item for sublist in con_A_cc_D5 for item in sublist]
con_B_cc_D5 = [item for sublist in con_B_cc_D5 for item in sublist]

con_A_cc_D9 = [item for sublist in con_A_cc_D9 for item in sublist]
con_B_cc_D9 = [item for sublist in con_B_cc_D9 for item in sublist]

labels = ['D0_A', 'D0_B', 'D1_A', 'D1_B', 'D5_A', 'D5_B', 'D9_A', 'D9_B']
raw = [con_A_cc_D0, con_B_cc_D0, con_A_cc_D1, con_B_cc_D1, con_A_cc_D5, con_B_cc_D5, con_A_cc_D9, con_B_cc_D9]

plt.figure(figsize=(10, 15))
plt.subplot(421); plt.hist(raw[0], bins=20, color='grey', alpha=0.4); plt.title('Th_D0'); plt.xlim((0,0.05))
#plt.subplot(422); plt.hist(raw[1], bins=20, color='b', alpha=0.4); plt.title('Th_D0_B'); plt.xlim((0,0.5))


plt.subplot(423); plt.hist(raw[2], bins=50, color='salmon', alpha=0.4); plt.title('Th_D1_A'); plt.xlim((0,0.05))
plt.subplot(424); plt.hist(raw[3], bins=50, color='turquoise', alpha=0.4); plt.title('Th_D1_B'); plt.xlim((0,0.05))

plt.subplot(425); plt.hist(raw[4], bins=50, color='salmon', alpha=0.4); plt.title('Th_D5_A');plt.xlim((0,0.05))
plt.subplot(426); plt.hist(raw[5], bins=50, color='turquoise', alpha=0.4); plt.title('Th_D5_B'); plt.xlim((0,0.05))

plt.subplot(427); plt.hist(raw[6], bins=50, color='salmon', alpha=0.4); plt.title('Th_D9_A'); plt.xlim((0,0.05))
plt.subplot(428); plt.hist(raw[7], bins=50, color='turquoise', alpha=0.4); plt.title('Th_D9_B'); plt.xlim((0,0.05))

plt.suptitle(f'Clustering coefficient [Pearson r val: {threshold}]')
plt.savefig(os.path.join(os.getcwd(), f"visualization/20210416/Th_clustering_coefficient.png"), dpi=300)
plt.show()