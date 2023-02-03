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

for count, mouse in enumerate(mouse_id_list):
    file_str = mouse[:len(mouse)-3]
    if file_str in treatment:
        group = 'Th'
    else:
        group = 'WT'
    day = mouse[len(mouse)-2:]
    if day == 'D5' or day == 'D0':
        continue
    else:
        if day == 'D1':
            event_day = 'Day1'
        else:
            event_day = 'Day9'
        data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,0:1800]
        random_data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')
        event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')

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
            x = random_nng.pearsons_correlation_matrix
            if not type(x) == np.float64:
                np.fill_diagonal(x, 0)
                x = np.nan_to_num(x)

                # Store event trace results
                results[count, 0] = np.median(x)
                results[count, 1] = np.mean(x)
                results[count, 2] = np.max(x)

                # Compute percentiles
                Q1 = np.percentile(x, 25, method='inverted_cdf')
                Q3 = np.percentile(x, 75, method='inverted_cdf')
                IQR = Q3 - Q1
                outlier_threshold = Q3 + 1.5 * IQR

                #ns_idx, A_idx, B_idx = nng.get_context_active(path_to_data + f'/{file_str}_{day}_neuron_context_active.csv')  # sort indices of context active cells
                ground_truth_nng_A = nng(data[context_selective_data, :])
                y = ground_truth_nng_A.pearsons_correlation_matrix
                np.fill_diagonal(y, 0)
                y = np.nan_to_num(y)

                # Store event trace results
                results[count, 3] = np.median(y)
                results[count, 4] = np.mean(y)
                results[count, 5] = np.max(y)




                random_vals = np.tril(x).flatten()
                data_vals = np.tril(y).flatten()
                test_result = stats.ks_2samp(random_vals, data_vals)
                # print(f"Binsize = 1 KS-statistic: {test_result}")
                if test_result.pvalue > 0.05:
                    print(f"{file_str} {day} not significant")
                else:
                    # Plot with threshold selected
                    plt.ylim(0, 200)
                    plt.hist(np.tril(x).flatten(), bins=50, color='grey', alpha=0.3)
                    plt.hist(np.tril(y).flatten(), bins=50, color='turquoise', alpha=0.3)
                    plt.axvline(x=outlier_threshold, color='red')
                    plt.legend(['threshold', 'shuffled', 'ground truth'])
                    plt.xlabel("Pearson's r-value")
                    plt.ylabel("Frequency")
                    plt.title(f'{group} {file_str} {day}: {tag}')
                    plt.savefig(EXPORT_PATH + f'{group}_{file_str}_{day}_{tag}_threshold_histogram_binned_event.png', dpi=300)
                    plt.show()


                #print(f"The threshold is: {outlier_threshold}")

                # Store event trace results
                results[count, 6] = outlier_threshold
                results[count, 7] = stats.ks_2samp(random_vals, data_vals)[1]


#%% EVENT ANALYSIS: Only looking at Context A (second half of data)
# store: mouse_id, mean, median, max pearson, threshold event separated, ks-stat event separated, mean, median, max pearson, threshold bin-separated, ks-stat bin-separated
results = np.zeros((58, 13))
set_cs =1 # 1 = context A, 0 = context B, -1 = nonselective

for count, mouse in enumerate(mouse_id_list):
    file_str = mouse[:len(mouse)-3]
    if file_str in treatment:
        group = 'Th'
    else:
        group = 'WT'
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
        random_data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,:]
        random_event_binned_data = generate_event_segmented(data=data.copy(), event_data=event_data)

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
            x = random_nng.pearsons_correlation_matrix
            if not type(x) == np.float64:
                np.fill_diagonal(x, 0)
                x = np.nan_to_num(x)

                # Store event trace results
                results[count, 0] = np.median(x)
                results[count, 1] = np.mean(x)
                results[count, 2] = np.max(x)

                # Compute percentiles
                Q1 = np.percentile(x, 25, method='inverted_cdf')
                Q3 = np.percentile(x, 75, method='inverted_cdf')
                IQR = Q3 - Q1
                outlier_threshold = Q3 + 1.5 * IQR

                #ns_idx, A_idx, B_idx = nng.get_context_active(path_to_data + f'/{file_str}_{day}_neuron_context_active.csv')  # sort indices of context active cells
                ground_truth_nng_A = nng(data[context_selective_data, :])
                y = ground_truth_nng_A.pearsons_correlation_matrix
                np.fill_diagonal(y, 0)
                y = np.nan_to_num(y)

                # Store event trace results
                results[count, 3] = np.median(y)
                results[count, 4] = np.mean(y)
                results[count, 5] = np.max(y)




                random_vals = np.tril(x).flatten()
                data_vals = np.tril(y).flatten()
                test_result = stats.ks_2samp(random_vals, data_vals)
                # print(f"Binsize = 1 KS-statistic: {test_result}")
                if test_result.pvalue > 0.05:
                    print(f"{file_str} {day} not significant")
                else:
                    # Plot with threshold selected
                    plt.ylim(0, 200)
                    plt.hist(np.tril(x).flatten(), bins=50, color='grey', alpha=0.3)
                    plt.hist(np.tril(y).flatten(), bins=50, color='salmon', alpha=0.3)
                    plt.axvline(x=outlier_threshold, color='red')
                    plt.legend(['threshold', 'shuffled', 'ground truth'])
                    plt.xlabel("Pearson's r-value")
                    plt.ylabel("Frequency")
                    plt.title(f'{group} {file_str} {day}: {tag}')
                    plt.savefig(EXPORT_PATH + f'{group}_{file_str}_{day}_{tag}_threshold_histogram_binned_event.png', dpi=300)
                    plt.show()


                #print(f"The threshold is: {outlier_threshold}")

                # Store event trace results
                results[count, 6] = outlier_threshold
                results[count, 7] = stats.ks_2samp(random_vals, data_vals)[1]