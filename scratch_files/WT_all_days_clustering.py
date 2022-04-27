"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 
File Final Edit Date:

Description: 
"""
"""
Created on April 15, 2021

@author: Veronica Porubsky

Title: Run batch analyses, collecting distributions

Context A - anxiogenic
Context B - neutral
"""
from dg_network_graph import DGNetworkGraph as nng
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
export_path = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/scratch_files/General_Exam/'
import_data_path = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/LC-DG-FC-data/'
dpi = 200

sns.set(style="whitegrid")

#%% Load untreated data files - WT
D1_WT = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv', '1055-3_D1_smoothed_calcium_traces.csv', '1055-4_D1_smoothed_calcium_traces.csv', '14-0_D1_smoothed_calcium_traces.csv', '122-1_D1_smoothed_calcium_traces.csv', '122-2_D1_smoothed_calcium_traces.csv', '122-3_D1_smoothed_calcium_traces.csv']#, '124-2_D1_smoothed_calcium_traces.csv']
D5_WT = ['1055-1_D5_smoothed_calcium_traces.csv', '1055-2_D5_smoothed_calcium_traces.csv', '1055-3_D5_smoothed_calcium_traces.csv', '1055-4_D5_smoothed_calcium_traces.csv', '14-0_D5_smoothed_calcium_traces.csv'] #'122-1_D5_smoothed_calcium_traces.csv', '122-2_D5_smoothed_calcium_traces.csv', '122-3_D5_smoothed_calcium_traces.csv', '124-2_D5_smoothed_calcium_traces.csv']
D9_WT = ['1055-1_D9_smoothed_calcium_traces.csv', '1055-2_D9_smoothed_calcium_traces.csv','1055-3_D9_smoothed_calcium_traces.csv', '1055-4_D9_smoothed_calcium_traces.csv', '14-0_D9_smoothed_calcium_traces.csv', '122-1_D9_smoothed_calcium_traces.csv', '122-2_D9_smoothed_calcium_traces.csv', '122-3_D9_smoothed_calcium_traces.csv']#, '124-2_D9_smoothed_calcium_traces.csv']
D0_WT = ['1055-1_D0_smoothed_calcium_traces.csv','1055-2_D0_smoothed_calcium_traces.csv','1055-3_D0_smoothed_calcium_traces.csv','1055-4_D0_smoothed_calcium_traces.csv','14-0_D0_smoothed_calcium_traces.csv']
all_WT_files = [D0_WT, D1_WT, D5_WT, D9_WT]

# %% All measurements, separating contexts
threshold = 0.3
names = []
data_mat = []



con_A_cc_D0 = []
con_B_cc_D0 = []
con_A_cc_D1 = []
con_B_cc_D1 = []
con_A_cc_D5 = []
con_B_cc_D5 = []
con_A_cc_D9 = []
con_B_cc_D9 = []

mouse_id_indices = []

# %% Context A and B
# Loop through all subjects and perform experimental and randomized network analyses
for day in [0, 1, 2, 3]:
    for mouse_id_index in range(len(all_WT_files[day])):
        filename = all_WT_files[day][mouse_id_index]
        mouse_id = filename.strip('_smoothed_calcium_traces.csv')

        if day == 0:
            mouse_id_indices.append(mouse_id.replace('_D0', ''))

        nn = nng(import_data_path+ filename)
        print(f"Executing analyses for {mouse_id}")
        num_neurons = nn.num_neurons

        # Context A and B graphs
        if day == 0:
            conA = nn.get_network_graph(threshold=threshold)
            conB = nn.get_network_graph(threshold=threshold)
        else:
            conA = nn.get_context_A_graph(threshold=threshold)
            conB = nn.get_context_B_graph(threshold=threshold)


        # clustering coefficient
        if day == 0:
            cc_A = nn.get_clustering_coefficient(threshold=threshold)
            cc_B = nn.get_clustering_coefficient(threshold=threshold)
        else:
            cc_A = nn.get_context_A_clustering_coefficient()
            cc_B = nn.get_context_B_clustering_coefficient()


        if day == 0:
            con_A_cc_D0.append(cc_A)
            con_B_cc_D0.append(cc_B)

        elif day == 1:
            con_A_cc_D1.append(cc_A)
            con_B_cc_D1.append(cc_B)

        elif day == 2:
            con_A_cc_D5.append(cc_A)
            con_B_cc_D5.append(cc_B)

        elif day == 3:
            con_A_cc_D9.append(cc_A)
            con_B_cc_D9.append(cc_B)

# %% clustering coefficient
con_A_cc_D0 = [item for sublist in con_A_cc_D0 for item in sublist]
con_B_cc_D0 = [item for sublist in con_B_cc_D0 for item in sublist]

con_A_cc_D1 = [item for sublist in con_A_cc_D1 for item in sublist]
con_B_cc_D1 = [item for sublist in con_B_cc_D1 for item in sublist]

con_A_cc_D5 = [item for sublist in con_A_cc_D5 for item in sublist]
con_B_cc_D5 = [item for sublist in con_B_cc_D5 for item in sublist]

con_A_cc_D9 = [item for sublist in con_A_cc_D9 for item in sublist]
con_B_cc_D9 = [item for sublist in con_B_cc_D9 for item in sublist]

#%%
labels = ['D0_A', 'D0_B', 'D1_A', 'D1_B', 'D5_A', 'D5_B', 'D9_A', 'D9_B']
raw = [con_A_cc_D0, con_B_cc_D0, con_A_cc_D1, con_B_cc_D1, con_A_cc_D5, con_B_cc_D5, con_A_cc_D9, con_B_cc_D9]

plt.figure(figsize=(10, 15))
plt.subplot(421); plt.hist(raw[0], bins=20, color='grey', alpha=0.4); plt.title('WT_D0')
#plt.subplot(422); plt.hist(raw[1], bins=20, color='b', alpha=0.4); plt.title('WT_D0_B'); plt.xlim((0,0.5))


plt.subplot(423); plt.hist(raw[2], bins=50, color='salmon', alpha=0.4); plt.title('WT_D1_A')
plt.subplot(424); plt.hist(raw[3], bins=50, color='turquoise', alpha=0.4); plt.title('WT_D1_B')

plt.subplot(425); plt.hist(raw[4], bins=50, color='salmon', alpha=0.4); plt.title('WT_D5_A')
plt.subplot(426); plt.hist(raw[5], bins=50, color='turquoise', alpha=0.4); plt.title('WT_D5_B')

plt.subplot(427); plt.hist(raw[6], bins=50, color='salmon', alpha=0.4); plt.title('WT_D9_A')
plt.subplot(428); plt.hist(raw[7], bins=50, color='turquoise', alpha=0.4); plt.title('WT_D9_B')

plt.suptitle(f'Clustering coefficient [Pearson r val: {threshold}]')
plt.savefig(export_path + 'WT_clustering_coefficient.png', dpi=300)
plt.show()


#%%
plt.figure(figsize=(10,5))
plt.subplot(131)
nn.plot_CDF_compare_two_samples(data_list=[con_A_cc_D0, con_A_cc_D1], color_list=['lightgrey', 'mistyrose'])
plt.legend(['D0', 'D1'])
plt.xlabel('Clustering coefficient')


plt.subplot(132)
nn.plot_CDF(data=con_A_cc_D0, color='lightgrey')
nn.plot_CDF(data=con_A_cc_D1, color='mistyrose')
nn.plot_CDF(data=con_A_cc_D9, color='salmon')
plt.ylabel('')
plt.xlabel('Clustering coefficient')
plt.title('Context A')

plt.subplot(133)
nn.plot_CDF(data=con_A_cc_D0, color='lightgrey')
nn.plot_CDF(data=con_B_cc_D1, color='turquoise')
nn.plot_CDF(data=con_B_cc_D9, color='teal')
plt.xlabel('Clustering coefficient')
plt.ylabel('')
plt.title('Context B')
plt.savefig(export_path + 'day0_day1_clustering.png', dpi=200)
plt.show()