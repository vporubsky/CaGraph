"""
Run correlated pairs ratio analysis and correlation coefficient analysis on
only the subset of cells that is matched across days 1 and 9.

Workflow:
1. Load matched cell-matched data
2. Create unique identifiers for each matched cell between days 1 and 9
3. Create networks as before
4. Run analyses as before
5. Pull out subset of data for cell-matched data
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

# %% clustering coefficient
path_to_data = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/LC-DG-FC-data/'
path_to_export = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/scratch_files/General_Exam/'
path_to_export = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/NAPE_talk/'
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
