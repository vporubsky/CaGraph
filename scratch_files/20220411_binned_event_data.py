"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: April 11, 2022
File Final Edit Date:

Description: Initial analysis of event data.
"""

import numpy as np
from neuronal_network_graph import DGNetworkGraph as nng
import matplotlib.pyplot as plt

# %% Global analysis parameters
threshold = 0.3
path_to_data = "/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/LC-DG-OFT-data/"

# %% Load untreated data files - saline

data_198_1 = ['198-1_Saline.csv', '198-1_Prop.csv', '198-1_Praz.csv', '198-1_Que.csv', '198-1_CNO.csv',
              '198-1_CNOSaline.csv', '198-1_CNOProp.csv', '198-1_CNOPraz.csv', '198-1_CNOQue.csv']

data_202_4 = ['202-4_Saline.csv', '202-4_Prop.csv', '202-4_Praz.csv', '202-4_Que.csv', '202-4_CNO.csv',
              '202-4_CNOSaline.csv', '202-4_CNOProp.csv', '202-4_CNOPraz.csv', '202-4_CNOQue.csv']

data_222_1 = ['222-1_Saline.csv', '222-1_Prop.csv', '222-1_Praz.csv', '222-1_Que.csv', '222-1_CNO.csv',
              '222-1_CNOSaline.csv', '222-1_CNOProp.csv', '222-1_CNOPraz.csv', '222-1_CNOQue.csv']

data_223_3 = ['223-3_Saline.csv', '223-3_Prop.csv', '223-3_Praz.csv', '223-3_Que.csv', '223-3_CNO.csv',
              '223-3_CNOSaline.csv', '223-3_CNOProp.csv', '223-3_CNOPraz.csv', '223-3_CNOQue.csv']

labels = ['Saline', 'Prop', 'Praz', 'Que 5mg/kg', 'CNO', 'CNO + Saline', 'CNO + Prop', 'CNO + Praz', 'CNO + Que']

#%%
# Function definitions
def get_indices(len_timeseries, interval):
    indices = []
    start_val = 0
    end_val = interval
    while end_val <= len_timeseries:
        indices.append((start_val, end_val))
        start_val = end_val
        end_val = start_val + interval
    return indices


def get_clustering_time_bins(data_path, list_of_datasets, indices, bin_size, threshold=0.3):
    """

    :param data_path:
    :param list_of_datasets:
    :return:
    """
    cc_tmp_array = []
    binned_event_count = []
    for dataset in list_of_datasets:

        mouse_id = dataset[0:5]

        try:
            binned_event_count.append(
                get_binned_event_traces(data_file=data_path + dataset.strip('.csv') + '_eventTrace.csv',
                                        bin_size=bin_size))
            nn = nng(data_path + dataset)
            print(f"Executing analyses for {mouse_id}")

            # Subsample graphs
            subsampled_graphs = nn.get_time_subsampled_graphs(subsample_indices=indices, threshold=threshold)

            cc_subsampled = []
            for graph in subsampled_graphs:
                # clustering coefficient
                cc = nn.get_clustering_coefficient(graph=graph)
                cc_subsampled.append(np.mean(cc))
            cc_tmp_array.append(cc_subsampled)

        except:
            print('No file found.')
            binned_event_count.append(0)
            cc_tmp_array.append([0])
    return cc_tmp_array, binned_event_count


def get_binned_event_traces(data_file, bin_size):
    data = np.genfromtxt(data_file, delimiter=",")[1:,:]
    np.shape(data)
    binned_data = np.zeros((np.shape(data)[0], int(np.shape(data)[1]/bin_size)))
    start_idx = 0
    for idx in range(np.shape(binned_data)[1]):
        binned_data[:, idx] = data[:, start_idx: start_idx + bin_size].sum(axis=1)
        start_idx = start_idx + bin_size
    return binned_data


#%%
data_file = data_198_1[0]
mouse_id = data_file[0:5]
bin_size = 6000
interval_size = 6000
indices = get_indices(len_timeseries=36000, interval=interval_size)

# Clustering Coefficient for each subject
cc_198_1, binned_events_198_1 = get_clustering_time_bins(data_path=path_to_data, list_of_datasets=data_198_1, indices=indices, bin_size=bin_size)
cc_202_4, binned_events_202_4 = get_clustering_time_bins(data_path=path_to_data, list_of_datasets=data_202_4, indices=indices, bin_size=bin_size)
cc_222_1, binned_events_222_1 = get_clustering_time_bins(data_path=path_to_data, list_of_datasets=data_222_1, indices=indices, bin_size=bin_size)
cc_223_3, binned_events_223_3 = get_clustering_time_bins(data_path=path_to_data, list_of_datasets=data_223_3, indices=indices, bin_size=bin_size)

#%% Plot averaged

for idx in range(np.shape(cc_223_3)[0]):
    plt.plot(np.mean(binned_events_223_3[idx], axis=0), cc_223_3[idx], 'k.')
plt.xticks([])
plt.yticks([])
plt.ylabel('Mean clustering coefficient')
plt.xlabel('Bin event count')
plt.show()
for idx in range(np.shape(cc_198_1)[0]):
    if idx != 4 and idx != 6:
        plt.plot(np.mean(binned_events_198_1[idx], axis=0), cc_198_1[idx], 'r.')
plt.xticks([])
plt.yticks([])
plt.ylabel('Mean clustering coefficient')
plt.xlabel('Bin event count')
plt.show()
for idx in range(np.shape(cc_202_4)[0]):
    plt.plot(np.mean(binned_events_202_4[idx], axis=0), cc_202_4[idx], 'b.')
plt.xticks([])
plt.yticks([])
plt.ylabel('Mean clustering coefficient')
plt.xlabel('Bin event count')
plt.show()
for idx in range(np.shape(cc_222_1)[0]):
    plt.plot(np.mean(binned_events_222_1[idx], axis=0), cc_222_1[idx], 'g.')
plt.xticks([])
plt.yticks([])
plt.ylabel('Mean clustering coefficient')
plt.xlabel('Bin event count')
plt.show()


#%% Plot by each neuron
nn = nng(path_to_data + '223-3_Saline.csv')