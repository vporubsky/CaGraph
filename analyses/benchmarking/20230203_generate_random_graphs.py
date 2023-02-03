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
            event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')
            random_event_binned_data = generate_event_segmented(data=data.copy(), event_data=event_data)
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
plt.savefig(EXPORT_PATH + f'{group}_ConB_{tag}_cdf_clustering.png', dpi=300)
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
            event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')
            random_event_binned_data = generate_event_segmented(data=data.copy(), event_data=event_data)
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
plt.savefig(EXPORT_PATH + f'{group}_ConA_{tag}_cdf_clustering.png', dpi=300)
plt.show()



#%% EVENT ANALYSIS: Only looking at Context B (first half of data)
# store: mouse_id, mean, median, max pearson, threshold event separated, ks-stat event separated, mean, median, max pearson, threshold bin-separated, ks-stat bin-separated
results = np.zeros((58, 13))
set_cs = 2 # 1 = context A, 0 = context B, -1 = nonselective

plt.figure(figsize=(10,10))
for count, mouse in enumerate(mouse_id_list):
    file_str = mouse[:len(mouse)-3]
    if file_str in treatment:
        group = 'Th'
        day = mouse[len(mouse)-2:]
        if day == 'D5' or day == 'D0':
            continue
        else:
            if day == 'D1':
                event_day = 'Day1'
            else:
                event_day = 'Day9'
            data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,0:1800]
            event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')
            random_event_binned_data = generate_event_segmented(data=data.copy(), event_data=event_data)
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
plt.savefig(EXPORT_PATH + f'{group}_ConB_{tag}_cdf_clustering.png', dpi=300)
plt.show()

#%% EVENT ANALYSIS: Only looking at Context A (second half of data)
# store: mouse_id, mean, median, max pearson, threshold event separated, ks-stat event separated, mean, median, max pearson, threshold bin-separated, ks-stat bin-separated
results = np.zeros((58, 13))
set_cs = 2 # 1 = context A, 0 = context B, -1 = nonselective

plt.figure(figsize=(10,10))
for count, mouse in enumerate(mouse_id_list):
    file_str = mouse[:len(mouse)-3]
    if file_str in treatment:
        group = 'Th'
        day = mouse[len(mouse)-2:]
        if day == 'D5' or day == 'D0':
            continue
        else:
            if day == 'D1':
                event_day = 'Day1'
            else:
                event_day = 'Day9'
            data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,1800:3600]
            event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')
            random_event_binned_data = generate_event_segmented(data=data.copy(), event_data=event_data)
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
plt.savefig(EXPORT_PATH + f'{group}_ConA_{tag}_cdf_clustering.png', dpi=300)
plt.show()



#%% hubs plotting
import plotting_utils
WT_gt_b = [8,0,6,0,1,4,8,3,3,0,11,0,3,0,3,0]
WT_r_b = [8,12,6,1,2,2,7,6,3,15,9,0,4,0,4,8]
WT_gt_a = [5,12,4,1,1,4,6,4,1,9,5,1,2,2,3]
WT_r_a = [7,11,5,0,0,0,10,5,5,11,9,1,5,3,0]
plotting_utils.plot_matched_data(WT_gt_b, WT_r_b, labels = ['WT gt b', 'WT r b'], colors=['turquoise', 'grey'])
plt.savefig(EXPORT_PATH + f'WT_Conb_hubs.png', dpi=300)
plt.show()
plotting_utils.plot_matched_data(WT_gt_a, WT_r_a, labels = ['WT gt a', 'WT r a'], colors=['salmon', 'grey'])
plt.savefig(EXPORT_PATH + f'WT_Cona_hubs.png', dpi=300)
plt.show()

Th_gt_b = [0,4,0,34,7,0,3,8,0,0,2,4,6,24,2,10]
Th_r_b = [0,5,5,16,8,3,4,10,0,1,6,3,8,14,2,9]
Th_gt_a = [5,3,7,3,1,3,0,2,4,1,1,2,12,3,2,4]
Th_r_a = [9,2,17,3,3,4,1,1,10,4,3,11,22,4,4,4]
plotting_utils.plot_matched_data(Th_gt_b, Th_r_b, labels = ['Th gt b', 'Th r b'], colors=['turquoise', 'grey'])
plt.savefig(EXPORT_PATH + f'Th_Conb_hubs.png', dpi=300)
plt.show()
plotting_utils.plot_matched_data(Th_gt_a, Th_r_a, labels = ['Th gt a', 'Th r a'], colors=['salmon', 'grey'])
plt.savefig(EXPORT_PATH + f'Th_Cona_hubs.png', dpi=300)
plt.show()