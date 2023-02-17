"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 
File Final Edit Date:

Description: This file is used to only analyze the context-selective cells


2023-02-13: using this file to troubleshoot why random correlations are so high
"""
# Import packages
from setup import FC_DATA_PATH
from dg_graph import DGGraph as nng
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
        day = mouse[len(mouse)-2:]
        if day == 'D5' or day == 'D0':
            continue
        else:
            if day == 'D1':
                event_day = 'Day1'
            else:
                event_day = 'Day9'
            data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,0:1800]
            random_data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,:] # Use when you want to shuffle full timeseries
            event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')
            ## Clean up neurons with one event
            data, event_data = remove_low_activity(data=data, event_data = event_data)
            print(np.shape(data))
            print(event_data.shape)
            random_event_binned_data = generate_event_segmented(data=data.copy(), event_data=event_data)
            data = data[:, 30:]
            random_event_binned_data = random_event_binned_data[:, 30:]
            #random_event_binned_data = random_event_binned_data[:,30:]
            #data = data[:, 30:]
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

                        #plt.hist(np.tril(x).flatten(), bins=50, color='grey', alpha=0.3)
                        random_nng.plot_CDF(data=np.tril(x).flatten(), color = 'grey')
                        random_nng.plot_CDF(data=np.tril(y).flatten(), color='turquoise')
                        #plt.hist(np.tril(y).flatten(), bins=50, color='turquoise', alpha=0.3)




                    #print(f"The threshold is: {outlier_threshold}")

                    # Store event trace results
                    results[count, 6] = outlier_threshold
                    results[count, 7] = stats.ks_2samp(random_vals, data_vals)[1]
            plt.legend(['shuffled', 'ground truth'])
            plt.xlabel("Pearson's r-value")
            plt.ylabel("Frequency")
            plt.title(f'{group}: {tag}')
            #plt.savefig(EXPORT_PATH + f'full_shuffle_{group}_ConB_{tag}_cdf_binned_event.png', dpi=300)
            plt.show()

#%% EVENT ANALYSIS: Only looking at Context A (second half of data)
# store: mouse_id, mean, median, max pearson, threshold event separated, ks-stat event separated, mean, median, max pearson, threshold bin-separated, ks-stat bin-separated
results = np.zeros((58, 13))

plt.figure(figsize= (10,10))
for count, mouse in enumerate(mouse_id_list):
    file_str = mouse[:len(mouse)-3]
    if file_str in WT:
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
                        # plt.hist(np.tril(x).flatten(), bins=50, color='grey', alpha=0.3)
                        random_nng.plot_CDF(data=np.tril(x).flatten(), color='grey')
                        random_nng.plot_CDF(data=np.tril(y).flatten(), color='salmon')
                        # plt.hist(np.tril(y).flatten(), bins=50, color='turquoise', alpha=0.3)
                        # plt.savefig(EXPORT_PATH + f'{group}_{file_str}_{day}_{tag}_threshold_histogram_binned_event.png', dpi=300)

                        # print(f"The threshold is: {outlier_threshold}")

                        # Store event trace results
                    results[count, 6] = outlier_threshold
                    results[count, 7] = stats.ks_2samp(random_vals, data_vals)[1]
plt.legend(['shuffled', 'ground truth'])
plt.xlabel("Pearson's r-value")
plt.ylabel("Frequency")
plt.title(f'{group}: {tag}')
plt.savefig(EXPORT_PATH + f'full_shuffle_{group}_ConA_{tag}_cdf_binned_event.png', dpi=300)
plt.show()




# Treatment group
# %% EVENT ANALYSIS: Only looking at Context B (first half of data)
# store: mouse_id, mean, median, max pearson, threshold event separated, ks-stat event separated, mean, median, max pearson, threshold bin-separated, ks-stat bin-separated
results = np.zeros((58, 13))

plt.figure(figsize=(10, 10))
for count, mouse in enumerate(mouse_id_list):
    file_str = mouse[:len(mouse) - 3]
    if file_str in treatment:
        group = 'Th'
        print(group)
        day = mouse[len(mouse) - 2:]
        if day == 'D5' or day == 'D0':
            continue
        else:
            if day == 'D1':
                event_day = 'Day1'
            else:
                event_day = 'Day9'
            data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,
                   0:1800]
            random_data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,:]
            event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')
            random_event_binned_data = generate_event_segmented(data=random_data.copy(), event_data=event_data)
            random_nng = nng(random_event_binned_data)
            random_ns_idx, random_A_idx, random_B_idx = random_nng.get_context_active(
                path_to_data + f'/{file_str}_{day}_neuron_context_active.csv')  # sort indices of context active cells
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
                context_selective_data = np.linspace(0, len(data) - 1, len(data), dtype=int)
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

                    # ns_idx, A_idx, B_idx = nng.get_context_active(path_to_data + f'/{file_str}_{day}_neuron_context_active.csv')  # sort indices of context active cells
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

                        # plt.hist(np.tril(x).flatten(), bins=50, color='grey', alpha=0.3)
                        random_nng.plot_CDF(data=np.tril(x).flatten(), color='grey')
                        random_nng.plot_CDF(data=np.tril(y).flatten(), color='turquoise')
                        # plt.hist(np.tril(y).flatten(), bins=50, color='turquoise', alpha=0.3)

                    # print(f"The threshold is: {outlier_threshold}")

                    # Store event trace results
                    results[count, 6] = outlier_threshold
                    results[count, 7] = stats.ks_2samp(random_vals, data_vals)[1]
plt.legend(['shuffled', 'ground truth'])
plt.xlabel("Pearson's r-value")
plt.ylabel("Frequency")
plt.title(f'{group}: {tag}')
plt.savefig(EXPORT_PATH + f'full_shuffle_{group}_ConB_{tag}_cdf_binned_event.png', dpi=300)
plt.show()

# %% EVENT ANALYSIS: Only looking at Context A (second half of data)
# store: mouse_id, mean, median, max pearson, threshold event separated, ks-stat event separated, mean, median, max pearson, threshold bin-separated, ks-stat bin-separated
results = np.zeros((58, 13))
plt.figure(figsize=(10, 10))
for count, mouse in enumerate(mouse_id_list):
    file_str = mouse[:len(mouse) - 3]
    if file_str in treatment:
        group = 'Th'
        day = mouse[len(mouse) - 2:]
        if day == 'D5' or day == 'D0':
            continue
        else:
            if day == 'D1':
                event_day = 'Day1'
            else:
                event_day = 'Day9'
            data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,
                   1800:3600]
            random_data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,:]
            event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')
            random_event_binned_data = generate_event_segmented(data=random_data.copy(), event_data=event_data)
            random_nng = nng(random_event_binned_data)
            random_ns_idx, random_A_idx, random_B_idx = random_nng.get_context_active(
                path_to_data + f'/{file_str}_{day}_neuron_context_active.csv')  # sort indices of context active cells
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
                context_selective_data = np.linspace(0, len(data) - 1, len(data), dtype=int)
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

                    # ns_idx, A_idx, B_idx = nng.get_context_active(path_to_data + f'/{file_str}_{day}_neuron_context_active.csv')  # sort indices of context active cells
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
                        # plt.hist(np.tril(x).flatten(), bins=50, color='grey', alpha=0.3)
                        random_nng.plot_CDF(data=np.tril(x).flatten(), color='grey')
                        random_nng.plot_CDF(data=np.tril(y).flatten(), color='salmon')
                        # plt.hist(np.tril(y).flatten(), bins=50, color='turquoise', alpha=0.3)
                        # plt.savefig(EXPORT_PATH + f'{group}_{file_str}_{day}_{tag}_threshold_histogram_binned_event.png', dpi=300)

                        # print(f"The threshold is: {outlier_threshold}")

                        # Store event trace results
                    results[count, 6] = outlier_threshold
                    results[count, 7] = stats.ks_2samp(random_vals, data_vals)[1]
plt.legend(['shuffled', 'ground truth'])
plt.xlabel("Pearson's r-value")
plt.ylabel("Frequency")
plt.title(f'{group}: {tag}')
plt.savefig(EXPORT_PATH + f'full_shuffle_{group}_ConA_{tag}_cdf_binned_event.png', dpi=300)
plt.show()



#%% Test pulling out specific neural traces to plot
time = random_event_binned_data[0,:]
for first_neuron in range(0, len(x)):
    for second_neuron in range(0, len(x)):
        if x[first_neuron, second_neuron] >= 0.7 and first_neuron != second_neuron:
            plt.plot(time, random_event_binned_data[first_neuron+1,:])
            plt.plot(time, random_event_binned_data[second_neuron+1,:])
            r, p = stats.pearsonr(random_event_binned_data[first_neuron+1, :], random_event_binned_data[second_neuron+1, :])
            plt.title(f"R-value: {r}")
            plt.show()


#%% Test pulling out specific neural traces to plot
time = data[0,:]
for first_neuron in range(0, len(y)):
    for second_neuron in range(0, len(y)):
        if y[first_neuron, second_neuron] >= 0.4:
            plt.plot(time, data[first_neuron+1,:])
            plt.plot(time, data[second_neuron+1,:])
            r, p = stats.pearsonr(data[first_neuron+1, :], data[second_neuron+1, :])
            plt.title(f"R-value: {r}")
            plt.show()
