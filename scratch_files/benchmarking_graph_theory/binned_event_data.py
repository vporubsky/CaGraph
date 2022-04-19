"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: April 11, 2022
File Final Edit Date:

Description: Initial analysis of event data. This file contains an analysis to plot the clustering coefficient of
each neuron vs. the number of binned events for a given time bin for that neuron, in order to determine if the highly
clustered neurons are primarily.

"""
import numpy as np
from neuronal_network_graph import DGNetworkGraph as nng
import matplotlib.pyplot as plt
import networkx as nx
import os

# %% Global analysis parameters
threshold = 0.3
path_to_data = os.getcwd() + "/LC-DG-FC-data/"

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
    dv_array = []
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
            dv_subsampled = []
            for graph in subsampled_graphs:
                # clustering coefficient
                cc = nn.get_clustering_coefficient(graph=graph)
                cc_subsampled.append(np.mean(cc))
                dv_subsampled.append(nx.clustering(graph))
            cc_tmp_array.append(cc_subsampled)
            dv_array.append(dv_subsampled)

        except:
            print('No file found.')
            binned_event_count.append(np.nan)
            cc_tmp_array.append([np.nan])
            dv_array.append([np.nan])

    return cc_tmp_array, dv_array,  binned_event_count


def get_binned_event_traces(data_file, bin_size):
    data = np.genfromtxt(data_file, delimiter=",")[1:,:]
    data = np.where(data < 1, data, 1) # This line will binarize the matrix
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
cc_198_1, dv_198_1, binned_events_198_1 = get_clustering_time_bins(data_path=path_to_data, list_of_datasets=data_198_1, indices=indices, bin_size=bin_size)
cc_202_4, dv_202_4, binned_events_202_4 = get_clustering_time_bins(data_path=path_to_data, list_of_datasets=data_202_4, indices=indices, bin_size=bin_size)
cc_222_1, dv_222_1, binned_events_222_1 = get_clustering_time_bins(data_path=path_to_data, list_of_datasets=data_222_1, indices=indices, bin_size=bin_size)
cc_223_3, dv_223_3, binned_events_223_3 = get_clustering_time_bins(data_path=path_to_data, list_of_datasets=data_223_3, indices=indices, bin_size=bin_size)

#%% Plot individual mice, averaged
import seaborn as sns
import matplotlib
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

fig, axes = plt.subplots(2,2, figsize=(20,20))
binned = []
cc = []
for idx in range(np.shape(cc_223_3)[0]):
    binned.append(np.mean(binned_events_223_3[idx], axis=0))
    cc.append(cc_223_3[idx])
binned = list(np.concatenate(binned).flat)
cc = list(np.concatenate(cc).flat)
axes[0,0].plot(binned, cc, 'k.')
sns.regplot(binned, cc, color ='blue', ax=axes[0,0])
axes[0,0].set_xticks([])
axes[0,0].set_yticks([])
axes[0,0].set_title('223-3')
axes[0,0].set_ylabel('Mean clustering coefficient')
axes[0,0].set_xlabel('Bin event count')

binned = []
cc = []
for idx in range(np.shape(cc_198_1)[0]):
    if idx != 4 and idx != 6:
        binned.append(np.mean(binned_events_198_1[idx], axis=0))
        cc.append(cc_198_1[idx])
binned = list(np.concatenate(binned).flat)
cc = list(np.concatenate(cc).flat)
axes[0,1].plot(binned, cc, 'k.')
sns.regplot(binned, cc, color ='blue', ax=axes[0,1])
axes[0,1].set_xticks([])
axes[0,1].set_yticks([])
axes[0,1].set_title('198-1')
axes[0,1].set_ylabel('Mean clustering coefficient')
axes[0,1].set_xlabel('Bin event count')


binned = []
cc = []
for idx in range(np.shape(cc_202_4)[0]):
    binned.append(np.mean(binned_events_202_4[idx], axis=0))
    cc.append(cc_202_4[idx])
binned = list(np.concatenate(binned).flat)
cc = list(np.concatenate(cc).flat)
axes[1,0].plot(binned, cc, 'k.')
sns.regplot(binned, cc, color ='blue', ax=axes[1,0])
axes[1,0].set_xticks([])
axes[1,0].set_yticks([])
axes[1,0].set_title('202-4')
axes[1,0].set_ylabel('Mean clustering coefficient')
axes[1,0].set_xlabel('Bin event count')


binned = []
cc = []
for idx in range(np.shape(cc_222_1)[0]):
    binned.append(np.mean(binned_events_222_1[idx], axis=0))
    cc.append(cc_222_1[idx])
binned = list(np.concatenate(binned).flat)
cc = list(np.concatenate(cc).flat)
axes[1,1].plot(binned, cc, 'k.')
sns.regplot(binned, cc, color ='blue', ax=axes[1,1])
axes[1,1].set_xticks([])
axes[1,1].set_yticks([])
axes[1,1].set_title('222-1')
axes[1,1].set_ylabel('Mean clustering coefficient')
axes[1,1].set_xlabel('Bin event count')

plt.savefig('mean_clustering_v_bin_event_count.png', dpi=200)
plt.show()


#%% Plot by each neuron -- 198-1
mouse_id = '198-1'

store_drug_condition_198_1 = []
# For each drug condition
for drug in range(len(dv_198_1)):

    # If the file does not exist, there will be a zero stored in binned_events_198_1[drug]
    if not type(binned_events_198_1[drug]) == np.ndarray:
        store_drug_condition_198_1.append(np.nan)
        continue

    # If the file exists, determine the cc for each neuron
    else:
        cc_per_neuron = np.zeros(np.shape(binned_events_198_1[drug]))
        x_vals = []
        for time_bin in range(len(dv_198_1[drug])):
            # For each neuron,
            for idx in range(np.shape(binned_events_198_1[drug])[0]):
                if binned_events_198_1[drug] is None:
                    continue
                else:
                    cc_val = dv_198_1[drug][time_bin][str(idx)]
                    bin_event_val = binned_events_198_1[drug][idx, time_bin]
                    x_vals.append(cc_val)
                    cc_per_neuron[idx, time_bin] = cc_val
    store_drug_condition_198_1.append(cc_per_neuron)

plt.figure(figsize=(10,10))
for drug in range(len(labels)):
    if not type(binned_events_198_1[drug]) == np.ndarray:
        continue
    else:
        for neuron in range(len(store_drug_condition_198_1[drug])):
            # average for each neuron across time bins
            plt.plot(np.mean(binned_events_198_1[drug][neuron]), np.mean(store_drug_condition_198_1[drug][neuron]), 'k.')
plt.xlabel('Mean binned events')
plt.ylabel('Mean clustering coefficient')
plt.title(mouse_id)
plt.savefig(os.getcwd() + f'/scratch_files/General_Exam/{mouse_id}_mean_cc_v_binned_events.png', dpi=200)
plt.show()


binned = []
cc = []
plt.figure(figsize=(10,10))
for drug in range(len(labels)):
    if not type(binned_events_198_1[drug]) == np.ndarray:
        continue
    else:
        for neuron in range(len(store_drug_condition_198_1[drug])):
            # average for each neuron across time bins
            plt.plot(binned_events_198_1[drug][neuron], store_drug_condition_198_1[drug][neuron], 'k.')
plt.xlabel('Num binned events')
plt.ylabel('Clustering coefficient')
plt.title(mouse_id)
plt.savefig(os.getcwd() + f'/scratch_files/General_Exam/{mouse_id}_cc_v_binned_events.png', dpi=200)
plt.show()

#%% Plot by each neuron -- 202-4
mouse_id = '202-4'

store_drug_condition_202_4 = []
# For each drug condition
for drug in range(len(dv_202_4)):

    # If the file does not exist, there will be a zero stored in binned_events_202_4[drug]
    if type(binned_events_202_4[drug]) == int:
        store_drug_condition_202_4.append(np.nan)
        continue

    # If the file exists, determine the cc for each neuron
    else:
        cc_per_neuron = np.zeros(np.shape(binned_events_202_4[drug]))
        x_vals = []
        for time_bin in range(len(dv_202_4[drug])):
            # For each neuron,
            for idx in range(np.shape(binned_events_202_4[drug])[0]):
                if binned_events_202_4[drug] is None:
                    continue
                else:
                    cc_val = dv_202_4[drug][time_bin][str(idx)]
                    bin_event_val = binned_events_202_4[drug][idx, time_bin]
                    x_vals.append(cc_val)
                    cc_per_neuron[idx, time_bin] = cc_val
    store_drug_condition_202_4.append(cc_per_neuron)

plt.figure(figsize=(10,10))
count = 0
for drug in range(len(labels)):
    if type(binned_events_202_4[drug]) is int:
        continue
        count += 1
    else:
        for neuron in range(len(store_drug_condition_202_4[drug])):
            # average for each neuron across time bins
            plt.plot(np.mean(binned_events_202_4[drug][neuron]), np.mean(store_drug_condition_202_4[drug][neuron]), 'k.')
plt.xlabel('Mean binned events')
plt.ylabel('Mean clustering coefficient')
plt.title(mouse_id)
plt.savefig(os.getcwd() + f'/scratch_files/General_Exam/{mouse_id}_mean_cc_v_binned_events.png', dpi=200)
plt.show()


plt.figure(figsize=(10,10))
# Plot each time bin - neuron pair
for drug in range(len(labels)):
    if type(binned_events_202_4[drug]) is int:
        continue
    else:
        for neuron in range(len(store_drug_condition_202_4[drug])):
            # average for each neuron across time bins
            plt.plot(binned_events_202_4[drug][neuron], store_drug_condition_202_4[drug][neuron], 'k.')
plt.xlabel('Num binned events')
plt.ylabel('Clustering coefficient')
plt.title(mouse_id)
plt.savefig(os.getcwd() + f'/scratch_files/General_Exam/{mouse_id}_cc_v_binned_events.png', dpi=200)
plt.show()


#%% Plot by each neuron -- 222-1

mouse_id = '222-1'

store_drug_condition_222_1 = []
# For each drug condition
for drug in range(len(dv_222_1)):

    # If the file does not exist, there will be a zero stored in binned_events_222_1[drug]
    if type(binned_events_222_1[drug]) == int:
        store_drug_condition_222_1.append(np.nan)
        continue

    # If the file exists, determine the cc for each neuron
    else:
        cc_per_neuron = np.zeros(np.shape(binned_events_222_1[drug]))
        x_vals = []
        for time_bin in range(len(dv_222_1[drug])):
            # For each neuron,
            for idx in range(np.shape(binned_events_222_1[drug])[0]):
                if binned_events_222_1[drug] is None:
                    continue
                else:
                    cc_val = dv_222_1[drug][time_bin][str(idx)]
                    bin_event_val = binned_events_222_1[drug][idx, time_bin]
                    x_vals.append(cc_val)
                    cc_per_neuron[idx, time_bin] = cc_val
    store_drug_condition_222_1.append(cc_per_neuron)

plt.figure(figsize=(10,10))
count = 0
for drug in range(len(labels)):
    if type(binned_events_222_1[drug]) is int:
        continue
        count += 1
    else:
        for neuron in range(len(store_drug_condition_222_1[drug])):
            # average for each neuron across time bins
            plt.plot(np.mean(binned_events_222_1[drug][neuron]), np.mean(store_drug_condition_222_1[drug][neuron]), 'k.')
plt.xlabel('Mean binned events')
plt.ylabel('Mean clustering coefficient')
plt.title(mouse_id)
plt.savefig(os.getcwd() + f'/scratch_files/General_Exam/{mouse_id}_mean_cc_v_binned_events.png', dpi=200)
plt.show()


plt.figure(figsize=(10,10))
# Plot each time bin - neuron pair
for drug in range(len(labels)):
    if type(binned_events_222_1[drug]) is int:
        continue
    else:
        for neuron in range(len(store_drug_condition_222_1[drug])):
            # average for each neuron across time bins
            plt.plot(binned_events_222_1[drug][neuron], store_drug_condition_222_1[drug][neuron], 'k.')
plt.xlabel('Num binned events')
plt.ylabel('Clustering coefficient')
plt.title(mouse_id)
plt.savefig(os.getcwd() + f'/scratch_files/General_Exam/{mouse_id}_cc_v_binned_events.png', dpi=200)
plt.show()

#%% Plot by each neuron -- 223-3
mouse_id = '223-3'

store_drug_condition_223_3 = []
# For each drug condition
for drug in range(len(dv_223_3)):

    # If the file does not exist, there will be a zero stored in binned_events_223_3[drug]
    if type(binned_events_223_3[drug]) == int:
        store_drug_condition_223_3.append(np.nan)
        continue

    # If the file exists, determine the cc for each neuron
    else:
        cc_per_neuron = np.zeros(np.shape(binned_events_223_3[drug]))
        x_vals = []
        for time_bin in range(len(dv_223_3[drug])):
            # For each neuron,
            for idx in range(np.shape(binned_events_223_3[drug])[0]):
                if binned_events_223_3[drug] is None:
                    continue
                else:
                    cc_val = dv_223_3[drug][time_bin][str(idx)]
                    bin_event_val = binned_events_223_3[drug][idx, time_bin]
                    x_vals.append(cc_val)
                    cc_per_neuron[idx, time_bin] = cc_val
    store_drug_condition_223_3.append(cc_per_neuron)

plt.figure(figsize=(10,10))
count = 0
for drug in range(len(labels)):
    if type(binned_events_223_3[drug]) is int:
        continue
        count += 1
    else:
        for neuron in range(len(store_drug_condition_223_3[drug])):
            # average for each neuron across time bins
            plt.plot(np.mean(binned_events_223_3[drug][neuron]), np.mean(store_drug_condition_223_3[drug][neuron]), 'k.')
plt.xlabel('Mean binned events')
plt.ylabel('Mean clustering coefficient')
plt.title(mouse_id)
plt.savefig(os.getcwd() + f'/scratch_files/General_Exam/{mouse_id}_mean_cc_v_binned_events.png', dpi=200)
plt.show()



# Plot each time bin - neuron pair
plt.figure(figsize=(10,10))
for drug in range(len(labels)):
    if type(binned_events_223_3[drug]) is int:
        continue
    else:
        for neuron in range(len(store_drug_condition_223_3[drug])):
            # average for each neuron across time bins
            plt.plot(binned_events_223_3[drug][neuron], store_drug_condition_223_3[drug][neuron], 'k.')
plt.xlabel('Num binned events')
plt.ylabel('Clustering coefficient')
plt.title(mouse_id)
plt.savefig(os.getcwd() + f'/scratch_files/General_Exam/{mouse_id}_cc_v_binned_events.png', dpi=200)
plt.show()


#%% Figure X
fig, axes = plt.subplots(2,2, figsize=(20,20))

mouse_id = '198-1'
binned = []
cc = []
for drug in range(len(labels)):
    if not type(binned_events_198_1[drug]) == np.ndarray:
        continue
    else:
        for neuron in range(len(store_drug_condition_198_1[drug])):

            binned.append(binned_events_198_1[drug][neuron])
            cc.append(store_drug_condition_198_1[drug][neuron])
binned = list(np.concatenate(binned).flat)
cc = list(np.concatenate(cc).flat)
axes[0,0].plot(binned, cc, 'k.')
axes[0,0].set_xticks([])
axes[0,0].set_yticks([])
axes[0,0].set_xlabel('Num binned events')
axes[0,0].set_ylabel('Clustering coefficient')
axes[0,0].set_title(mouse_id)

mouse_id = '202-4'
binned = []
cc = []
for drug in range(len(labels)):
    if type(binned_events_202_4[drug]) is int:
        continue
    else:
        for neuron in range(len(store_drug_condition_202_4[drug])):
            binned.append(binned_events_202_4[drug][neuron])
            cc.append(store_drug_condition_202_4[drug][neuron])
binned = list(np.concatenate(binned).flat)
cc = list(np.concatenate(cc).flat)
axes[0,1].plot(binned, cc, 'k.')
axes[0,1].set_xticks([])
axes[0,1].set_yticks([])
axes[0,1].set_xlabel('Num binned events')
axes[0,1].set_ylabel('Clustering coefficient')
axes[0,1].set_title(mouse_id)

mouse_id = '222-1'
binned = []
cc = []
for drug in range(len(labels)):
    if type(binned_events_222_1[drug]) is int:
        continue
    else:
        for neuron in range(len(store_drug_condition_222_1[drug])):
            binned.append(binned_events_222_1[drug][neuron])
            cc.append(store_drug_condition_222_1[drug][neuron])
binned = list(np.concatenate(binned).flat)
cc = list(np.concatenate(cc).flat)
axes[1,0].set_xticks([])
axes[1,0].set_yticks([])
axes[1,0].plot(binned, cc, 'k.')
axes[1,0].set_xlabel('Num binned events')
axes[1,0].set_ylabel('Clustering coefficient')
axes[1,0].set_title(mouse_id)

mouse_id = '223-3'
binned = []
cc = []
for drug in range(len(labels)):
    if type(binned_events_223_3[drug]) is int:
        continue
    else:
        for neuron in range(len(store_drug_condition_223_3[drug])):
            binned.append(binned_events_223_3[drug][neuron])
            cc.append(store_drug_condition_223_3[drug][neuron])
binned = list(np.concatenate(binned).flat)
cc = list(np.concatenate(cc).flat)
axes[1,1].plot(binned, cc, 'k.')
axes[1,1].set_xticks([])
axes[1,1].set_yticks([])
axes[1,1].set_xlabel('Num binned events')
axes[1,1].set_ylabel('Clustering coefficient')
axes[1,1].set_title(mouse_id)
plt.savefig(os.getcwd() + f"/scratch_files/General_Exam/cc_v_binned_events.png", dpi=200)
plt.show()

#%% Given a neuron label, plot the time-course for that time bin
