"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: December 2, 2021
File Final Edit Date:

Description: An initial investigation of the OFT dataset under several pharmacologic interventions.

Experimental set-up:
- Recordings taken from the DG in the OFT
- Drug conditions: saline, propanolol, prazosin, quetiapine (5mgkg, 10mg/kg),
                    CNO + Saline, CNO + prazosin, CNO + quetiapine
- Propanolol: beta blocker, reduces anxiety, inhibits effects of norepinephrine
- Prazosin: alpha-1 blocker receptor for norepinephrine
- Quetiapine: antipsychotic medicine, thought to block dopamine type 2 and serotonin 2A receptors
- CNO: activates DREADDS

First dataset animals:
- 198-1
- 202-4

Second dataset:
- 222-1
- 223-3

Context A and B:
- Context A: pre-drug
- Context B: post-drug

Experimental conclusions/ notes:
- low number of experimental replicates
- 198-1 died before CNO and CNO + Propranolol conditions could be run

"""
from neuronal_network_graph import DGNetworkGraph as nng
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import os


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


def get_clustering_time_bins(data_path, list_of_datasets, indices, threshold=0.3):
    """

    :param data_path:
    :param list_of_datasets:
    :return:
    """
    cc_store_means = []
    cc_tmp_array = np.zeros((np.shape(list_of_datasets)[0] * np.shape(list_of_datasets)[1], len(indices)))
    array_idx = 0
    for dataset in list_of_datasets:
        for ca_file in dataset:
            mouse_id = ca_file[0:5]
            try:
                nn = nng(data_path + ca_file)
                print(f"Executing analyses for {mouse_id}")
                num_neurons = nn.num_neurons

                # Subsample graphs
                subsampled_graphs = nn.get_time_subsampled_graphs(subsample_indices=indices, threshold=threshold)

                cc_tmp = []
                cc_tmp_mean = []
                for graph in subsampled_graphs:
                    # clustering coefficient
                    cc = nn.get_clustering_coefficient(graph=graph)
                    cc_tmp.append(cc)
                    cc_tmp_mean.append(np.mean(cc))
                cc_store_means.append(np.mean(cc_tmp))

                cc_tmp_array[array_idx, :] = cc_tmp_mean

            except:
                print('No file found.')
                cc_store_means.append('Nan')
                cc_tmp_array[array_idx, :] = 0

            array_idx += 1
    return cc_tmp_array


def get_clustering_contexts(data_path, list_of_datasets, indices, threshold=0.3):
    """
    Samples time-series into

    :param data_path:
    :param list_of_datasets:
    :return:
    """
    cc_A = []
    cc_B = []
    for ca_file in list_of_datasets:
        mouse_id = ca_file[0:5]
        try:
            nn = nng(data_path + ca_file)
            print(f"Executing analyses for {mouse_id}")

            # Subsample graphs
            subsampled_graphs = nn.get_time_subsampled_graphs(subsample_indices=indices, threshold=threshold)

            for idx, graph in enumerate(subsampled_graphs):
                # clustering coefficient
                cc = nn.get_clustering_coefficient(graph=graph)
                cc_B_tmp = []
                if idx == 0:
                    # let the first graph be context A (pre-drug)
                    cc_A.append(cc)
                else:
                    # else, concatenate all of the remaining time bins together for context B (post-drug)
                    cc_B_tmp = cc_B_tmp + cc
            cc_B.append(cc_B_tmp)

        except:
            cc_A.append([0])
            cc_B.append([0])

    return cc_A, cc_B


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

# %%
# Clustering Coefficient for each subject with updated context A, B
indices = get_indices(len_timeseries=36000, interval=6000)
cc_A_198_1, cc_B_198_1 = get_clustering_contexts(data_path=path_to_data, list_of_datasets=data_198_1, indices=indices)
cc_A_202_4, cc_B_202_4 = get_clustering_contexts(data_path=path_to_data, list_of_datasets=data_202_4, indices=indices)
cc_A_222_1, cc_B_222_1 = get_clustering_contexts(data_path=path_to_data, list_of_datasets=data_222_1, indices=indices)
cc_A_223_3, cc_B_223_3 = get_clustering_contexts(data_path=path_to_data, list_of_datasets=data_223_3, indices=indices)

# %% Figure 1: 198-1
plt.figure(figsize=(15, 10))
for idx in range(len(labels)):
    # CDF prestim, stim ----------------------------------------------
    stat_lev = stats.ks_2samp(cc_A_198_1[idx], cc_B_198_1[idx])

    # sort the data in ascending order
    x = np.sort(cc_A_198_1[idx])
    # get the cdf values of y
    y = np.arange(len(cc_A_198_1[idx])) / float(len(cc_A_198_1[idx]))

    # plotting
    plt.subplot(250 + idx + 1)
    plt.plot(x, y, 'salmon', marker='o')

    # sort the data in ascending order
    x = np.sort(cc_B_198_1[idx])
    # get the cdf values of y
    y = np.arange(len(cc_B_198_1[idx])) / float(len(cc_B_198_1[idx]))

    # plotting
    plt.subplot(250 + idx + 1)
    plt.plot(x, y, 'turquoise', marker='o')

    plt.title(labels[idx])

plt.suptitle('198-1')
plt.savefig('fig_1_198-1_clustering.png', dpi=300)
plt.show()

# %% Figure 2: 202-4
plt.figure(figsize=(15, 10))
for idx in range(len(labels)):
    # CDF prestim, stim ----------------------------------------------
    stat_lev = stats.ks_2samp(cc_A_202_4[idx], cc_B_202_4[idx])

    # sort the data in ascending order
    x = np.sort(cc_A_202_4[idx])
    # get the cdf values of y
    y = np.arange(len(cc_A_202_4[idx])) / float(len(cc_A_202_4[idx]))

    # plotting
    plt.subplot(250 + idx + 1)
    plt.plot(x, y, 'salmon', marker='o')

    # sort the data in ascending order
    x = np.sort(cc_B_202_4[idx])
    # get the cdf values of y
    y = np.arange(len(cc_B_202_4[idx])) / float(len(cc_B_202_4[idx]))

    # plotting
    plt.subplot(250 + idx + 1)
    plt.plot(x, y, 'turquoise', marker='o')

    plt.title(labels[idx])

plt.suptitle('202-4')
plt.savefig('fig_2_202-4_clustering.png', dpi=300)
plt.show()

# %% Figure 3: 222-1
plt.figure(figsize=(15, 10))
for idx in range(len(labels)):
    # CDF prestim, stim ----------------------------------------------
    stat_lev = stats.ks_2samp(cc_A_222_1[idx], cc_B_222_1[idx])

    # sort the data in ascending order
    x = np.sort(cc_A_222_1[idx])
    # get the cdf values of y
    y = np.arange(len(cc_A_222_1[idx])) / float(len(cc_A_222_1[idx]))

    # plotting
    plt.subplot(250 + idx + 1)
    plt.plot(x, y, 'salmon', marker='o')

    # sort the data in ascending order
    x = np.sort(cc_B_222_1[idx])
    # get the cdf values of y
    y = np.arange(len(cc_B_222_1[idx])) / float(len(cc_B_222_1[idx]))

    # plotting
    plt.subplot(250 + idx + 1)
    plt.plot(x, y, 'turquoise', marker='o')

    plt.title(labels[idx])

plt.suptitle('222-1')
plt.savefig('fig_3_222-1_clustering.png', dpi=300)
plt.show()

# %% Figure 4: 223-3
plt.figure(figsize=(15, 10))
for idx in range(len(labels)):
    # CDF prestim, stim ----------------------------------------------
    stat_lev = stats.ks_2samp(cc_A_223_3[idx], cc_B_223_3[idx])

    # sort the data in ascending order
    x = np.sort(cc_A_223_3[idx])
    # get the cdf values of y
    y = np.arange(len(cc_A_223_3[idx])) / float(len(cc_A_223_3[idx]))

    # plotting
    plt.subplot(250 + idx + 1)
    plt.plot(x, y, 'salmon', marker='o')

    # sort the data in ascending order
    x = np.sort(cc_B_223_3[idx])
    # get the cdf values of y
    y = np.arange(len(cc_B_223_3[idx])) / float(len(cc_B_223_3[idx]))

    # plotting
    plt.subplot(250 + idx + 1)
    plt.plot(x, y, 'turquoise', marker='o')

    plt.title(labels[idx])

plt.suptitle('223-3')
plt.savefig('fig_4_223-3_clustering.png', dpi=300)
plt.show()

# %% Figure 5:
import scipy
plt.figure(figsize=(20, 10))

for idx in range(len(labels)):
    mean_cc_A_202_4 = np.mean(cc_A_202_4[idx])
    mean_cc_B_202_4 = np.mean(cc_B_202_4[idx])

    mean_cc_A_198_1 = np.mean(cc_A_198_1[idx])
    mean_cc_B_198_1 = np.mean(cc_B_198_1[idx])

    mean_cc_A_221_1 = np.mean(cc_A_222_1[idx])
    mean_cc_B_221_1 = np.mean(cc_B_222_1[idx])

    mean_cc_A_223_3 = np.mean(cc_A_223_3[idx])
    mean_cc_B_223_3 = np.mean(cc_B_223_3[idx])

    A = [mean_cc_A_198_1, mean_cc_A_202_4, mean_cc_A_221_1, mean_cc_A_223_3]
    B = [mean_cc_B_198_1, mean_cc_B_202_4, mean_cc_B_221_1, mean_cc_B_223_3]
    ts, p = scipy.stats.ttest_rel(A,B)
    print(f"{labels[idx]} t-test p-value: {p}")
    # plotting
    plt.subplot(250 + idx + 1)
    plt.plot(0, mean_cc_A_198_1, 'salmon', marker='o')
    plt.plot(0, mean_cc_A_202_4, 'salmon', marker='o')
    plt.plot(0, mean_cc_A_221_1, 'salmon', marker='o')
    plt.plot(0, mean_cc_A_223_3, 'salmon', marker='o')

    plt.plot(1, mean_cc_B_198_1, 'turquoise', marker='o')
    plt.plot(1, mean_cc_B_202_4, 'turquoise', marker='o')
    plt.plot(1, mean_cc_B_221_1, 'turquoise', marker='o')
    plt.plot(1, mean_cc_B_223_3, 'turquoise', marker='o')
    plt.plot([0, 1], [mean_cc_A_198_1, mean_cc_B_198_1], 'lightgrey')
    plt.plot([0, 1], [mean_cc_A_202_4, mean_cc_B_202_4], 'lightgrey')
    plt.plot([0, 1], [mean_cc_A_221_1, mean_cc_B_221_1], 'lightgrey')
    plt.plot([0, 1], [mean_cc_A_223_3, mean_cc_B_223_3], 'lightgrey')

    plt.title(labels[idx])
    plt.ylim(-0.05, 0.51)
    plt.xlim(-0.5, 1.5)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4])
    plt.xticks([])

plt.suptitle('')
plt.savefig('fig_5_mean_clustering.png', dpi=300)
plt.show()


# %% More subsamples
# Todo: add generate indices function to nng class
list_of_datasets = [data_198_1, data_202_4, data_222_1, data_223_3]

indices = get_indices(len_timeseries=36000, interval=6000)
cc_tmp_array = get_clustering_time_bins(data_path=path_to_data, list_of_datasets=list_of_datasets, indices=indices)

# %% Make pandas dataframe to export Excel file of binned data
pd_indices = ['198-1 Saline', '198-1 Prop', '198-1 Praz', '198-1 Que 5mg/kg', '198-1 CNO', '198-1 CNO + Saline',
              '198-1 CNO + Prop', '198-1 CNO + Praz', '198-1 CNO + Que',
              '202-4 Saline', '202-4 Prop', '202-4 Praz', '202-4 Que 5mg/kg', '202-4 CNO', '202-4 CNO + Saline',
              '202-4 CNO + Prop', '202-4 CNO + Praz', '202-4 CNO + Que',
              '222-1 Saline', '222-1 Prop', '222-1 Praz', '222-1 Que 5mg/kg', '222-1 CNO', '222-1 CNO + Saline',
              '222-1 CNO + Prop', '222-1 CNO + Praz', '222-1 CNO + Que',
              '223-3 Saline', '223-3 Prop', '223-3 Praz', '223-3 Que 5mg/kg', '223-3 CNO', '223-3 CNO + Saline',
              '223-3 CNO + Prop', '223-3 CNO + Praz', '223-3 CNO + Que']
pd_labels = ['0-10min', '10-20min', '20-30min', '30-40min', '40-50min', '50-60min']
df = pd.DataFrame(cc_tmp_array, columns=pd_labels, index=pd_indices)

df.to_excel('OFT_binned_data_mean_clustering_coefficient_6000_tp_bins.xlsx')

# %%
import matplotlib
labels_leg = ['Saline', 'Prop', 'Praz', 'Que 5mg/kg', 'CNO', 'CNO + Saline', 'CNO + Prop', 'CNO + Praz', 'CNO + Que']

fig = plt.figure(figsize=(25, 20))
font = {'size': 12}

matplotlib.rc('font', **font)

for i, (name, row) in enumerate(df.iterrows()):
    if i < 9:
        ax = plt.subplot(411)
        ax.plot(row, '.-')
        plt.ylabel('cc')
        plt.legend(labels_leg, bbox_to_anchor=(1.04, 1), loc="upper left")
    elif i >= 9 and i < 18:
        ax = plt.subplot(412)
        ax.plot(row, '.-')
        plt.ylabel('cc')
    elif i >= 18 and i < 27:
        ax = plt.subplot(413)
        ax.plot(row, '.-')
        plt.ylabel('cc')
    else:
        ax = plt.subplot(414)
        ax.plot(row, '.-')
        plt.ylabel('cc')

plt.savefig("fig_9_mean_clustering_time_binned_6000_tp_bins.png", bbox_inches="tight", dpi=300)
plt.show()

# %% Figure 11: averages the 5 post-drug time bins
font = {'size': 16}

matplotlib.rc('font', **font)
plt.figure(figsize=(20, 10))
for idx in range(len(labels)):
    mean_cc_A_202_4 = np.mean(cc_tmp_array[9 + idx, 0])
    mean_cc_B_202_4 = np.mean(cc_tmp_array[9 + idx, 1:])

    mean_cc_A_198_1 = np.mean(cc_tmp_array[0 + idx, 0])
    mean_cc_B_198_1 = np.mean(cc_tmp_array[0 + idx, 1:])

    mean_cc_A_221_1 = np.mean(cc_tmp_array[18 + idx, 0])
    mean_cc_B_221_1 = np.mean(cc_tmp_array[18 + idx, 1:])

    mean_cc_A_223_3 = np.mean(cc_tmp_array[27 + idx, 0])
    mean_cc_B_223_3 = np.mean(cc_tmp_array[27 + idx, 1:])

    # plotting
    plt.subplot(250 + idx + 1)
    plt.plot(0, mean_cc_A_198_1, 'salmon', marker='o')
    plt.plot(0, mean_cc_A_202_4, 'salmon', marker='o')
    plt.plot(0, mean_cc_A_221_1, 'salmon', marker='o')
    plt.plot(0, mean_cc_A_223_3, 'salmon', marker='o')

    plt.plot(1, mean_cc_B_198_1, 'turquoise', marker='o')
    plt.plot(1, mean_cc_B_202_4, 'turquoise', marker='o')
    plt.plot(1, mean_cc_B_221_1, 'turquoise', marker='o')
    plt.plot(1, mean_cc_B_223_3, 'turquoise', marker='o')

    plt.plot([0, 1], [mean_cc_A_198_1, mean_cc_B_198_1], 'lightgrey')
    plt.plot([0, 1], [mean_cc_A_202_4, mean_cc_B_202_4], 'lightgrey')
    plt.plot([0, 1], [mean_cc_A_221_1, mean_cc_B_221_1], 'lightgrey')
    plt.plot([0, 1], [mean_cc_A_223_3, mean_cc_B_223_3], 'lightgrey')

    plt.title(labels[idx])
    plt.ylim(-0.05, 0.41)
    plt.xlim(-0.5, 1.5)
    plt.yticks([0, 0.1, 0.2, 0.3, 0.4])
    plt.xticks([])

plt.suptitle('')
plt.savefig('fig_11_mean_clustering.png', dpi=300)
plt.show()

# %% Figure 12:
font = {'size': 12}

labels = ['Saline', 'Prop', 'Praz', 'Que5mg-kg', 'CNO', 'CNOSaline', 'CNOProp', 'CNOPraz', 'CNOQue']
labels = ['Saline', 'Prop', 'Praz', 'Que 5mg/kg', 'CNO', 'CNO + Saline', 'CNO + Prop', 'CNO + Praz', 'CNO + Que']
plt.figure(figsize=(20,10))
matplotlib.rc('font', **font)
for idx in range(len(labels)):

    # plotting
    plt.subplot(250 + idx + 1)
    plt.plot(cc_tmp_array[9 + idx, :], 'grey', marker='o')
    plt.plot(cc_tmp_array[0 + idx, :], 'grey', marker='o')
    plt.plot(cc_tmp_array[18 + idx, :], 'grey', marker='o')
    plt.plot(cc_tmp_array[27 + idx, :], 'grey', marker='o')


    plt.title(labels[idx])
    plt.yticks([])
    plt.xticks([])

plt.suptitle('')
plt.savefig(f'fig_12_mean_clustering.png', dpi=300)
plt.show()



#%% plot mean cc vs. num neurons
y1 = [mean_cc_A_202_4, mean_cc_A_198_1, mean_cc_A_223_3, mean_cc_A_221_1]
y2 = [mean_cc_B_202_4, mean_cc_B_198_1, mean_cc_B_223_3, mean_cc_B_221_1]
x = [218,139,44,53]
z = np.polyfit(x,y1,1)
p = np.poly1d(z)
plt.scatter(x, y1, c='salmon', marker='.')
plt.scatter(x, y2, c='turquoise', marker='.')
plt.plot(x, p(x), c='grey')
plt.xlabel('number of neurons')
plt.ylabel('mean clustering coefficient')
plt.savefig('cc_v_num_neurons.png', dpi = 200)
plt.plot()
plt.show()

#%% transitivity
nn_198 = nng(path_to_data)