"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 
File Final Edit Date:

Description: This file is used to only analyze the context-selective cells
"""
# Import packages
import plotting_utils
from setup import FC_DATA_PATH
from dg_network_graph import DGNetworkGraph as nng
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import os
from scipy import stats

path_to_data = FC_DATA_PATH
EXPORT_PATH = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/analyses/benchmarking/random_networks/scratch-analysis/'

treatment = ['2-1', '2-2', '2-3', '348-1', '349-2', '386-2', '387-4', '396-1']
WT = ['1055-1', '1055-2', '1055-3', '1055-4', '14-0', '122-1', '122-2', '122-3', '124-2']

#%% assemble list of ids
mouse_id_list = []
for file in os.listdir(path_to_data):
    if file.endswith('_smoothed_calcium_traces.csv'):
        print(file)
        mouse_id = file.replace('_smoothed_calcium_traces.csv', '')
        print(mouse_id)
        mouse_id_list.append(mouse_id)

#%% EVENT ANALYSIS: Only looking at Context B (first half of data)
# store: mouse_id, mean, median, max pearson, threshold event separated, ks-stat event separated, mean, median, max pearson, threshold bin-separated, ks-stat bin-separated
results = np.zeros((58, 13))
set_cs = 2 # 1 = context A, 0 = context B, -1 = nonselective

plt.figure(figsize=(10,10))
for count, mouse in enumerate(mouse_id_list):
    file_str = mouse[:len(mouse)-3]
    if file_str in WT:
        group = 'WT'
        print(group)
        day = mouse[len(mouse)-2:]
        if day == 'D5' or day == 'D0':
            continue
        else:
            if day == 'D1':
                event_day = 'Day1'
            else:
                event_day = 'Day9'
            data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,0:1800]
            random_data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,:]
            event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')
            random_event_binned_data = generate_event_segmented(data=random_data.copy(), event_data=event_data)
            random_nng = nng(random_event_binned_data)
            random_ns_idx, random_A_idx, random_B_idx = random_nng.get_context_active(path_to_data + f'/{file_str}_{day}_neuron_context_active.csv')  # sort indices of context active cells
            if set_cs == 1:
                context_selective_data = random_A_idx
                tag = 'context A selective'
            elif set_cs == 0:
                context_selective_data = random_B_idx
                tag = 'context B selective'
            elif set_cs == -1:
                context_selective_data = random_ns_idx
                tag = 'non-selective'
            else:
                context_selective_data = np.linspace(0, len(data)-1, len(data), dtype=int)
                tag = 'all cells'
            if len(context_selective_data) > 1:
                random_nng = nng(random_event_binned_data[context_selective_data, :])
                x = random_nng.get_clustering_coefficient(threshold=0.3)


                #ns_idx, A_idx, B_idx = nng.get_context_active(path_to_data + f'/{file_str}_{day}_neuron_context_active.csv')  # sort indices of context active cells
                ground_truth_nng_A = nng(data[context_selective_data, :])
                y = ground_truth_nng_A.get_clustering_coefficient(threshold=0.3)

                test_result = stats.ks_2samp(x, y)
                # print(f"Binsize = 1 KS-statistic: {test_result}")
                if test_result.pvalue > 0.05:
                    print(f"{file_str} {day} not significant")
                else:
                        # Plot with threshold selected

                        #plt.hist(np.tril(x).flatten(), bins=50, color='grey', alpha=0.3)
                    random_nng.plot_CDF(data=np.tril(x).flatten(), color = 'grey')
                    ground_truth_nng_A.plot_CDF(data=np.tril(y).flatten(), color='turquoise')

            ground_truth_hubs = len(random_nng.get_hubs()[0])
            random_hubs = len(ground_truth_nng_A.get_hubs()[0])
            print(f"gt hubs: {ground_truth_hubs}, random hubs: {random_hubs}")




plt.legend(['shuffled', 'ground truth'])
plt.xlabel("Clustering Coefficient")
plt.ylabel("CDF")
plt.title(f'{group}: {tag}')
plt.savefig(EXPORT_PATH + f'full_timeseries_randomized_{group}_ConB_{tag}_cdf_clustering.png', dpi=300)
plt.show()

#%% EVENT ANALYSIS: Only looking at Context A (second half of data)
# store: mouse_id, mean, median, max pearson, threshold event separated, ks-stat event separated, mean, median, max pearson, threshold bin-separated, ks-stat bin-separated
results = np.zeros((58, 13))
set_cs = 2 # 1 = context A, 0 = context B, -1 = nonselective

plt.figure(figsize=(10,10))
for count, mouse in enumerate(mouse_id_list):
    file_str = mouse[:len(mouse)-3]
    if file_str in WT:
        group = 'WT'
        print(group)
        day = mouse[len(mouse)-2:]
        if day == 'D5' or day == 'D0':
            continue
        else:
            if day == 'D1':
                event_day = 'Day1'
            else:
                event_day = 'Day9'
            data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,1800:3600]
            random_data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,:]
            event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')
            random_event_binned_data = generate_event_segmented(data=random_data.copy(), event_data=event_data)
            random_nng = nng(random_event_binned_data)
            random_ns_idx, random_A_idx, random_B_idx = random_nng.get_context_active(path_to_data + f'/{file_str}_{day}_neuron_context_active.csv')  # sort indices of context active cells
            if set_cs == 1:
                context_selective_data = random_A_idx
                tag = 'context A selective'
            elif set_cs == 0:
                context_selective_data = random_B_idx
                tag = 'context B selective'
            elif set_cs == -1:
                context_selective_data = random_ns_idx
                tag = 'non-selective'
            else:
                context_selective_data = np.linspace(0, len(data)-1, len(data), dtype=int)
                tag = 'all cells'
            if len(context_selective_data) > 1:
                random_nng = nng(random_event_binned_data[context_selective_data, :])
                x = random_nng.get_clustering_coefficient(threshold=0.3)


                #ns_idx, A_idx, B_idx = nng.get_context_active(path_to_data + f'/{file_str}_{day}_neuron_context_active.csv')  # sort indices of context active cells
                ground_truth_nng_A = nng(data[context_selective_data, :])
                y = ground_truth_nng_A.get_clustering_coefficient(threshold=0.3)

                test_result = stats.ks_2samp(x, y)
                # print(f"Binsize = 1 KS-statistic: {test_result}")
                if test_result.pvalue > 0.05:
                    print(f"{file_str} {day} not significant")
                else:
                        # Plot with threshold selected

                        #plt.hist(np.tril(x).flatten(), bins=50, color='grey', alpha=0.3)
                    random_nng.plot_CDF(data=np.tril(x).flatten(), color = 'grey')
                    ground_truth_nng_A.plot_CDF(data=np.tril(y).flatten(), color='salmon')

            ground_truth_hubs = len(random_nng.get_hubs()[0])
            random_hubs = len(ground_truth_nng_A.get_hubs()[0])
            print(f"gt hubs: {ground_truth_hubs}, random hubs: {random_hubs}")



plt.legend(['shuffled', 'ground truth'])
plt.xlabel("Clustering Coefficient")
plt.ylabel("CDF")
plt.title(f'{group}: {tag}')
plt.savefig(EXPORT_PATH + f'full_timeseries_randomized_{group}_ConA_{tag}_cdf_clustering.png', dpi=300)
plt.show()

