"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 
File Final Edit Date:

Description: 
"""
from dg_graph import DGGraph as nng
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy
import matplotlib
matplotlib.rcParams.update({'font.size': 14})
#%% Data files
# Wildtype condition
D1_WT = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv',
         '1055-4_D1_smoothed_calcium_traces.csv',
         '122-1_D1_smoothed_calcium_traces.csv',
         '122-2_D1_smoothed_calcium_traces.csv', '122-3_D1_smoothed_calcium_traces.csv', '14-0_D1_smoothed_calcium_traces.csv']
D9_WT = ['1055-1_D9_smoothed_calcium_traces.csv', '1055-2_D9_smoothed_calcium_traces.csv',
         '1055-4_D9_smoothed_calcium_traces.csv',
         '14-0_D9_smoothed_calcium_traces.csv', '122-1_D9_smoothed_calcium_traces.csv',
         '122-2_D9_smoothed_calcium_traces.csv', '122-3_D9_smoothed_calcium_traces.csv']

# Treatment condition
D1_Th = ['348-1_D1_smoothed_calcium_traces.csv',
         '349-2_D1_smoothed_calcium_traces.csv',
         '387-4_D1_smoothed_calcium_traces.csv', '396-1_D1_smoothed_calcium_traces.csv',
         '396-3_D1_smoothed_calcium_traces.csv']
D9_Th = ['387-4_D9_smoothed_calcium_traces.csv', '396-1_D9_smoothed_calcium_traces.csv',
         '396-3_D9_smoothed_calcium_traces.csv', '348-1_D9_smoothed_calcium_traces.csv','349-2_D9_smoothed_calcium_traces.csv']

# %% set hyper-parameters
threshold = 0.2

#%% ##################################### clustering coefficient with cell-matched data #####################################
path_to_data = '/LC-DG-FC-data/'
path_to_export = '/NAPE_talk/'
day = 'D1'
condition = 'WT'
condition_data = D1_WT
plt.figure(figsize=(15,15))
cc_all_mice_1_A = []
cc_all_mice_1_B = []
cc_all_mice_9_A = []
cc_all_mice_9_B = []
idx = 1
for filename in condition_data:
    mouse_id = filename.replace('_' + day + '_smoothed_calcium_traces.csv', '')
    path = os.getcwd() + '/LC-DG-FC-data/cell_matching_data/'
    file = mouse_id + '_cellRegistered.csv'
    path_to_file = path + file
    data = np.genfromtxt(path_to_file, delimiter=",")

    # delete rows with zeros
    data = data[~np.any(data == 0, axis=1)]

    # decrement all indices by 1 to convert Matlab indexing to Python
    data = data - np.ones(np.shape(data))

    nn_D1 = nng(path_to_data + mouse_id + '_D1_smoothed_calcium_traces.csv')
    nn_D9 = nng(path_to_data + mouse_id + '_D9_smoothed_calcium_traces.csv')

    nn_D1_con_A = nn_D1.get_context_A_graph(threshold=threshold)
    nn_D1_con_B = nn_D1.get_context_B_graph(threshold=threshold)

    nn_D9_con_A = nn_D9.get_context_A_graph(threshold=threshold)
    nn_D9_con_B = nn_D9.get_context_B_graph(threshold=threshold)

    #
    # clustering coefficient
    nn_D1_con_A_cc = nn_D1.get_context_A_clustering_coefficient()
    nn_D1_con_B_cc = nn_D1.get_context_B_clustering_coefficient()

    # correlated pairs ratio
    nn_D1_con_A_cr = nn_D1.get_context_A_correlated_pair_ratio(threshold=threshold)
    nn_D1_con_B_cr = nn_D1.get_context_B_correlated_pair_ratio(threshold=threshold)

    # clustering coefficient
    nn_D9_con_A_cc = nn_D9.get_context_A_clustering_coefficient()
    nn_D9_con_B_cc = nn_D9.get_context_B_clustering_coefficient()

    # correlated pairs ratio
    nn_D9_con_A_cr = nn_D9.get_context_A_correlated_pair_ratio(threshold=threshold)
    nn_D9_con_B_cr = nn_D9.get_context_B_correlated_pair_ratio(threshold=threshold)

    # Day 1, context A
    cc_1_A = []
    cr_1_A = []
    clustering = nx.clustering(nn_D1_con_A)
    corr_pair = nx.degree(nn_D1_con_A)
    [cc_1_A.append(clustering[str(int(node))]) for node in data[:, 0]]
    [cr_1_A.append(corr_pair[str(int(node))] / nn_D1.num_neurons) for node in data[:, 0]]

    # Day 9, context A
    cc_9_A = []
    cr_9_A = []
    clustering = nx.clustering(nn_D9_con_A)
    corr_pair = nx.degree(nn_D9_con_A)
    [cc_9_A.append(clustering[str(int(node))]) for node in data[:, 1]]
    [cr_9_A.append(corr_pair[str(int(node))] / nn_D9.num_neurons) for node in data[:, 1]]

    # Day 1, context B
    cc_1_B = []
    cr_1_B = []
    clustering = nx.clustering(nn_D1_con_B)
    corr_pair = nx.degree(nn_D1_con_B)
    [cc_1_B.append(clustering[str(int(node))]) for node in data[:, 0]]
    [cr_1_B.append(corr_pair[str(int(node))] / nn_D1.num_neurons) for node in data[:, 0]]

    # Day 9, context B
    cc_9_B = []
    cr_9_B = []
    clustering = nx.clustering(nn_D9_con_B)
    corr_pair = nx.degree(nn_D9_con_B)
    [cc_9_B.append(clustering[str(int(node))]) for node in data[:, 1]]
    [cr_9_B.append(corr_pair[str(int(node))] / nn_D9.num_neurons) for node in data[:, 1]]

    idx += 1

    # create lists with all clustering coefficient data to plot matched samples
    cc_all_mice_1_A = cc_all_mice_1_A + cc_1_A
    cc_all_mice_1_B = cc_all_mice_1_B + cc_1_B
    cc_all_mice_9_A = cc_all_mice_9_A + cc_9_A
    cc_all_mice_9_B = cc_all_mice_9_B + cc_9_B


#%% Context A data
set1 = cc_all_mice_1_A
set2 = cc_all_mice_9_A

# Put into dataframe
df = pd.DataFrame({'D1': set1, 'D9': set2})
data = pd.melt(df)

plt.figure(figsize=(15,15))
# Plot
fig, ax = plt.subplots()
sns.swarmplot(data=data, x='variable', y='value', ax=ax, size=0)

# Now connect the dots
# Find idx0 and idx1 by inspecting the elements return from ax.get_children()
idx0 = 0
idx1 = 1
locs1 = ax.get_children()[idx0].get_offsets()
locs2 = ax.get_children()[idx1].get_offsets()

y_all = np.zeros(2)
for i in range(locs1.shape[0]):
    x = [locs1[i, 0], locs2[i, 0]]
    y = [locs1[i, 1], locs2[i, 1]]
    ax.plot(x, y, color='lightgrey', linewidth=0.5)
    ax.plot(locs1[i, 0], locs1[i, 1], '.', color='salmon')
    ax.plot(locs2[i, 0], locs2[i, 1], '.', color='salmon')
    data = [locs1[:, 1], locs2[:, 1]]
    ax.boxplot(data, positions=[0,1], capprops =  dict(linewidth=0.5, color = 'salmon'),
               whiskerprops = dict(linewidth=0.5, color = 'salmon'),
               boxprops = dict(linewidth=0.5, color = 'salmon'),
               medianprops=dict(color='salmon'))
    plt.xticks([])
    y_all = np.vstack((y_all,y))

plt.xlabel(f'P-value = {scipy.stats.ttest_rel(set1, set2).pvalue:.3}')
plt.yticks([0.5, 1])
plt.ylabel('')
plt.savefig(path_to_export + f'{condition}_clustering_conA_matched.png', transparent=True, dpi=300)
plt.show()

#%% Context B data
set1 = cc_all_mice_1_B
set2 = cc_all_mice_9_B

# Put into dataframe
df = pd.DataFrame({'D1': set1, 'D9': set2})
data = pd.melt(df)

plt.figure(figsize=(15,15))
# Plot
fig, ax = plt.subplots()
sns.swarmplot(data=data, x='variable', y='value', ax=ax, size=0)

# Now connect the dots
# Find idx0 and idx1 by inspecting the elements return from ax.get_children()
idx0 = 0
idx1 = 1
locs1 = ax.get_children()[idx0].get_offsets()
locs2 = ax.get_children()[idx1].get_offsets()

y_all = np.zeros(2)
for i in range(locs1.shape[0]):
    x = [locs1[i, 0], locs2[i, 0]]
    y = [locs1[i, 1], locs2[i, 1]]
    ax.plot(x, y, color='lightgrey', linewidth=0.5)
    ax.plot(locs1[i, 0], locs1[i, 1], '.', color='turquoise')
    ax.plot(locs2[i, 0], locs2[i, 1], '.', color='turquoise')
    data = [locs1[:, 1], locs2[:, 1]]
    ax.boxplot(data, positions=[0, 1], capprops=dict(linewidth=0.5, color='turquoise'),
               whiskerprops=dict(linewidth=0.5, color='turquoise'),
               boxprops=dict(linewidth=0.5, color='turquoise'),
               medianprops=dict(color='turquoise'))
    plt.xticks([])
    y_all = np.vstack((y_all, y))

plt.xlabel(f'P-value = {scipy.stats.ttest_rel(set1, set2).pvalue:.3}')
plt.yticks([0.5, 1])
plt.ylabel('')

plt.savefig(path_to_export + f'/{condition}_clustering_conB_matched.png', transparent=True, dpi=300)
plt.show()

print(scipy.stats.ttest_rel(set1, set2))


# %% clustering coefficient
day = 'D1'
condition = 'Th'
condition_data = D1_Th
plt.figure(figsize=(15,15))
cc_all_mice_1_A = []
cc_all_mice_1_B = []
cc_all_mice_9_A = []
cc_all_mice_9_B = []
idx = 1
for filename in condition_data:
    mouse_id = filename.replace('_' + day + '_smoothed_calcium_traces.csv', '')
    path = os.getcwd() + '/LC-DG-FC-data/cell_matching_data/'
    file = mouse_id + '_cellRegistered.csv'
    path_to_file = path + file
    data = np.genfromtxt(path_to_file, delimiter=",")

    # delete rows with zeros
    data = data[~np.any(data == 0, axis=1)]

    # decrement all indices by 1 to convert Matlab indexing to Python
    data = data - np.ones(np.shape(data))

    nn_D1 = nng(path_to_data + mouse_id + '_D1_smoothed_calcium_traces.csv')
    nn_D9 = nng(path_to_data + mouse_id + '_D9_smoothed_calcium_traces.csv')

    nn_D1_con_A = nn_D1.get_context_A_graph(threshold=threshold)
    nn_D1_con_B = nn_D1.get_context_B_graph(threshold=threshold)

    nn_D9_con_A = nn_D9.get_context_A_graph(threshold=threshold)
    nn_D9_con_B = nn_D9.get_context_B_graph(threshold=threshold)

    #
    # clustering coefficient
    nn_D1_con_A_cc = nn_D1.get_context_A_clustering_coefficient()
    nn_D1_con_B_cc = nn_D1.get_context_B_clustering_coefficient()

    # correlated pairs ratio
    nn_D1_con_A_cr = nn_D1.get_context_A_correlated_pair_ratio(threshold=threshold)
    nn_D1_con_B_cr = nn_D1.get_context_B_correlated_pair_ratio(threshold=threshold)

    # clustering coefficient
    nn_D9_con_A_cc = nn_D9.get_context_A_clustering_coefficient()
    nn_D9_con_B_cc = nn_D9.get_context_B_clustering_coefficient()

    # correlated pairs ratio
    nn_D9_con_A_cr = nn_D9.get_context_A_correlated_pair_ratio(threshold=threshold)
    nn_D9_con_B_cr = nn_D9.get_context_B_correlated_pair_ratio(threshold=threshold)

    # Day 1, context A
    cc_1_A = []
    cr_1_A = []
    clustering = nx.clustering(nn_D1_con_A)
    corr_pair = nx.degree(nn_D1_con_A)
    [cc_1_A.append(clustering[str(int(node))]) for node in data[:, 0]]
    [cr_1_A.append(corr_pair[str(int(node))] / nn_D1.num_neurons) for node in data[:, 0]]

    # Day 9, context A
    cc_9_A = []
    cr_9_A = []
    clustering = nx.clustering(nn_D9_con_A)
    corr_pair = nx.degree(nn_D9_con_A)
    [cc_9_A.append(clustering[str(int(node))]) for node in data[:, 1]]
    [cr_9_A.append(corr_pair[str(int(node))] / nn_D9.num_neurons) for node in data[:, 1]]

    # Day 1, context B
    cc_1_B = []
    cr_1_B = []
    clustering = nx.clustering(nn_D1_con_B)
    corr_pair = nx.degree(nn_D1_con_B)
    [cc_1_B.append(clustering[str(int(node))]) for node in data[:, 0]]
    [cr_1_B.append(corr_pair[str(int(node))] / nn_D1.num_neurons) for node in data[:, 0]]

    # Day 9, context B
    cc_9_B = []
    cr_9_B = []
    clustering = nx.clustering(nn_D9_con_B)
    corr_pair = nx.degree(nn_D9_con_B)
    [cc_9_B.append(clustering[str(int(node))]) for node in data[:, 1]]
    [cr_9_B.append(corr_pair[str(int(node))] / nn_D9.num_neurons) for node in data[:, 1]]

    idx += 1

    # create lists with all clustering coefficient data to plot matched samples
    cc_all_mice_1_A = cc_all_mice_1_A + cc_1_A
    cc_all_mice_1_B = cc_all_mice_1_B + cc_1_B
    cc_all_mice_9_A = cc_all_mice_9_A + cc_9_A
    cc_all_mice_9_B = cc_all_mice_9_B + cc_9_B



#%% Context A data
set1 = cc_all_mice_1_A
set2 = cc_all_mice_9_A

# Put into dataframe
df = pd.DataFrame({'D1': set1, 'D9': set2})
data = pd.melt(df)

plt.figure(figsize=(15,15))
# Plot
fig, ax = plt.subplots()
sns.swarmplot(data=data, x='variable', y='value', ax=ax, size=0)

# Now connect the dots
# Find idx0 and idx1 by inspecting the elements return from ax.get_children()
idx0 = 0
idx1 = 1
locs1 = ax.get_children()[idx0].get_offsets()
locs2 = ax.get_children()[idx1].get_offsets()

y_all = np.zeros(2)
for i in range(locs1.shape[0]):
    x = [locs1[i, 0], locs2[i, 0]]
    y = [locs1[i, 1], locs2[i, 1]]
    ax.plot(x, y, color='lightgrey', linewidth=0.5)
    ax.plot(locs1[i, 0], locs1[i, 1], '.', color='salmon')
    ax.plot(locs2[i, 0], locs2[i, 1], '.', color='salmon')
    data = [locs1[:, 1], locs2[:, 1]]
    ax.boxplot(data, positions=[0, 1], capprops=dict(linewidth=0.5, color='salmon'),
               whiskerprops=dict(linewidth=0.5, color='salmon'),
               boxprops=dict(linewidth=0.5, color='salmon'),
               medianprops=dict(color='salmon'))
    plt.xticks([])
    y_all = np.vstack((y_all, y))

plt.xlabel(f'P-value = {scipy.stats.ttest_rel(set1, set2).pvalue:.3}')
plt.yticks([0.5, 1])
plt.ylabel('')
plt.savefig(path_to_export + f'{condition}_clustering_conA_matched.png', transparent=True, dpi=300)
plt.show()

print(scipy.stats.ttest_rel(set1, set2))

#%% Context B data
set1 = cc_all_mice_1_B
set2 = cc_all_mice_9_B

# Put into dataframe
df = pd.DataFrame({'D1': set1, 'D9': set2})
data = pd.melt(df)

plt.figure(figsize=(15,15))
# Plot
fig, ax = plt.subplots()
sns.swarmplot(data=data, x='variable', y='value', ax=ax, size=0)

# Now connect the dots
# Find idx0 and idx1 by inspecting the elements return from ax.get_children()
idx0 = 0
idx1 = 1
locs1 = ax.get_children()[idx0].get_offsets()
locs2 = ax.get_children()[idx1].get_offsets()


y_all = np.zeros(2)
for i in range(locs1.shape[0]):
    x = [locs1[i, 0], locs2[i, 0]]
    y = [locs1[i, 1], locs2[i, 1]]
    ax.plot(x, y, color='lightgrey', linewidth=0.5)
    ax.plot(locs1[i, 0], locs1[i, 1], '.', color='turquoise')
    ax.plot(locs2[i, 0], locs2[i, 1], '.', color='turquoise')
    data = [locs1[:, 1], locs2[:, 1]]
    ax.boxplot(data, positions=[0, 1], capprops=dict(linewidth=0.5, color='turquoise'),
               whiskerprops=dict(linewidth=0.5, color='turquoise'),
               boxprops=dict(linewidth=0.5, color='turquoise'),
               medianprops=dict(color='turquoise'))
    plt.xticks([])
    y_all = np.vstack((y_all, y))

plt.xlabel(f'P-value = {scipy.stats.ttest_rel(set1, set2).pvalue:.3}')
plt.yticks([0.5, 1])
plt.ylabel('')
plt.savefig(path_to_export + f'{condition}_clustering_conB_matched.png', transparent=True, dpi=300)
plt.show()

print(scipy.stats.ttest_rel(set1, set2))



#%% ##################################### clustering coefficient vs baseline #####################################
# %% All measurements, separating contexts
#%% Load untreated data files - WT
D1_WT = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv', '1055-3_D1_smoothed_calcium_traces.csv', '1055-4_D1_smoothed_calcium_traces.csv', '14-0_D1_smoothed_calcium_traces.csv']# '122-1_D1_smoothed_calcium_traces.csv', '122-2_D1_smoothed_calcium_traces.csv', '122-3_D1_smoothed_calcium_traces.csv']#, '124-2_D1_smoothed_calcium_traces.csv']
D5_WT = ['1055-1_D5_smoothed_calcium_traces.csv', '1055-2_D5_smoothed_calcium_traces.csv', '1055-3_D5_smoothed_calcium_traces.csv', '1055-4_D5_smoothed_calcium_traces.csv', '14-0_D5_smoothed_calcium_traces.csv'] #'122-1_D5_smoothed_calcium_traces.csv', '122-2_D5_smoothed_calcium_traces.csv', '122-3_D5_smoothed_calcium_traces.csv', '124-2_D5_smoothed_calcium_traces.csv']
D9_WT = ['1055-1_D9_smoothed_calcium_traces.csv', '1055-2_D9_smoothed_calcium_traces.csv','1055-3_D9_smoothed_calcium_traces.csv', '1055-4_D9_smoothed_calcium_traces.csv', '14-0_D9_smoothed_calcium_traces.csv']# '122-1_D9_smoothed_calcium_traces.csv', '122-2_D9_smoothed_calcium_traces.csv', '122-3_D9_smoothed_calcium_traces.csv']#, '124-2_D9_smoothed_calcium_traces.csv']
D0_WT = ['1055-1_D0_smoothed_calcium_traces.csv','1055-2_D0_smoothed_calcium_traces.csv','1055-3_D0_smoothed_calcium_traces.csv','1055-4_D0_smoothed_calcium_traces.csv','14-0_D0_smoothed_calcium_traces.csv']
all_WT_files = [D0_WT, D1_WT, D5_WT, D9_WT]
threshold = 0.25
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

        nn = nng(path_to_data + filename)
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
            con_A_cc_D0 += [cc_A]
            con_B_cc_D0 += [cc_B]

        elif day == 1:
            con_A_cc_D1 += [cc_A]
            con_B_cc_D1 += [cc_B]

        elif day == 2:
            con_A_cc_D5 += [cc_A]
            con_B_cc_D5 += [cc_B]

        elif day == 3:
            con_A_cc_D9 += [cc_A]
            con_B_cc_D9 += [cc_B]




#%%
idx = 1
plt.figure(figsize=(10,5))
plt.subplot(131)
nn.plot_CDF_compare_two_samples(data_list=[con_A_cc_D0[idx], con_A_cc_D1[idx]], color_list=['lightgrey', 'mistyrose'])
plt.legend(['D0', 'D1'])
plt.xlabel('Clustering coefficient')


plt.subplot(132)
nn.plot_CDF(data=con_A_cc_D0[idx], color='lightgrey')
nn.plot_CDF(data=con_A_cc_D1[idx], color='mistyrose')
nn.plot_CDF(data=con_A_cc_D9[idx], color='salmon')
plt.ylabel('')
plt.xlabel('Clustering coefficient')
plt.title('Context A')

plt.subplot(133)
nn.plot_CDF(data=con_A_cc_D0[idx], color='lightgrey')
nn.plot_CDF(data=con_B_cc_D1[idx], color='turquoise')
nn.plot_CDF(data=con_B_cc_D9[idx], color='teal')
plt.xlabel('Clustering coefficient')
plt.ylabel('')
plt.title('Context B')
plt.savefig(path_to_export + 'day0_day1_clustering.png', dpi=200)
plt.show()