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
from neuronal_network_graph import DGNetworkGraph as nng
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
threshold = 0.3

# %% clustering coefficient
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
    path = os.getcwd() + '/cell_matching_data/'
    file = mouse_id + '_cellRegistered.csv'
    path_to_file = path + file
    data = np.genfromtxt(path_to_file, delimiter=",")

    # delete rows with zeros
    data = data[~np.any(data == 0, axis=1)]

    # decrement all indices by 1 to convert Matlab indexing to Python
    data = data - np.ones(np.shape(data))

    nn_D1 = nng(mouse_id + '_D1_smoothed_calcium_traces.csv')
    nn_D9 = nng(mouse_id + '_D9_smoothed_calcium_traces.csv')

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


    # Build dataframe

    palette = {"D1_A": "mistyrose", "D9_A": "salmon", "D1_B": "paleturquoise", "D9_B":"darkturquoise"}

    labels = ['D1_A', 'D9_A', 'D1_B', 'D9_B']
    raw = [cc_1_A, cc_9_A, cc_1_B, cc_9_B]
    data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels)

    plt.subplot(3,3,idx)
    sns.swarmplot(data=data, color = 'k');
    sns.boxplot(data=data, whis = 1.5, palette=palette);

    plt.title(mouse_id)

    idx += 1

    # create lists with all clustering coefficient data to plot matched samples
    cc_all_mice_1_A = cc_all_mice_1_A + cc_1_A
    cc_all_mice_1_B = cc_all_mice_1_B + cc_1_B
    cc_all_mice_9_A = cc_all_mice_9_A + cc_9_A
    cc_all_mice_9_B = cc_all_mice_9_B + cc_9_B
plt.suptitle(f'{condition} Clustering coefficient cell-matched, threshold: {threshold}')
plt.savefig(os.path.join(os.getcwd(), f"visualization/20210603/{condition}_clustering_cell_matched.png"), transparent=True, dpi=300)
plt.show()


# %% correlated pairs
plt.figure(figsize=(15,15))
idx = 0
for filename in condition_data:
    idx += 1
    mouse_id = filename.replace('_' + day + '_smoothed_calcium_traces.csv', '')
    path = os.getcwd() + '/cell_matching_data/'
    file = mouse_id + '_cellRegistered.csv'
    path_to_file = path + file
    data = np.genfromtxt(path_to_file, delimiter=",")

    # delete rows with zeros
    data = data[~np.any(data == 0, axis=1)]

    # decrement all indices by 1 to convert Matlab indexing to Python
    data = data - np.ones(np.shape(data))

    nn_D1 = nng(mouse_id + '_D1_smoothed_calcium_traces.csv')
    nn_D9 = nng(mouse_id + '_D9_smoothed_calcium_traces.csv')

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


    # Build dataframe

    palette = {"D1_A": "mistyrose", "D9_A": "salmon", "D1_B": "paleturquoise", "D9_B":"darkturquoise"}

    labels = ['D1_A', 'D9_A', 'D1_B', 'D9_B']
    raw = [cr_1_A, cr_9_A, cr_1_B, cr_9_B]
    data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels)

    plt.subplot(3,3,idx)
    sns.swarmplot(data=data, color = 'k');
    sns.boxplot(data=data, whis = 1.5, palette=palette);

    plt.title(mouse_id)

plt.suptitle(f'{condition} Correlated pairs ratio cell-matched, threshold: {threshold}')
plt.savefig(os.path.join(os.getcwd(), f"visualization/20210603/{condition}_corr_pair_matched.png"), transparent=True, dpi=300)
plt.show()


#%% Context A data
set1 = cc_all_mice_1_A
set2 = cc_all_mice_9_A

# Put into dataframe
df = pd.DataFrame({'D1': set1, 'D9': set2})
data = pd.melt(df)

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
    ax.plot(x, y, color='salmon', alpha=0.2)
    y_all = np.vstack((y_all,y))

ave_slope = np.mean(y_all[1:,1] - y_all[1:,0])
plt.xlabel('')
plt.ylabel('clustering coeff.')
plt.plot([0, 1], [0.5-ave_slope/2, 0.5+ave_slope/2], 'k')

plt.suptitle(f'{condition} Context A')
plt.savefig(os.path.join(os.getcwd(), f"visualization/20210603/{condition}_clustering_conA_matched.png"), transparent=True, dpi=300)
plt.show()

print(scipy.stats.ttest_rel(set1, set2))

#%% Context B data
set1 = cc_all_mice_1_B
set2 = cc_all_mice_9_B

# Put into dataframe
df = pd.DataFrame({'D1': set1, 'D9': set2})
data = pd.melt(df)

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
    ax.plot(x, y, color='turquoise', alpha=0.2)
    y_all = np.vstack((y_all, y))

ave_slope = np.mean(y_all[1:, 1] - y_all[1:, 0])
plt.xlabel('')
plt.ylabel('clustering coeff.')
plt.plot([0, 1], [0.5 - ave_slope / 2, 0.5 + ave_slope / 2], 'k')
plt.suptitle(f'{condition} Context B')
plt.savefig(os.path.join(os.getcwd(), f"visualization/20210603/{condition}_clustering_conB_matched.png"), transparent=True, dpi=300)
plt.show()

print(scipy.stats.ttest_rel(set1, set2))

# %% Context A data
set1 = cc_all_mice_1_A
set2 = cc_all_mice_9_A

# Put into dataframe
df = pd.DataFrame({'D1': set1, 'D9': set2})
data = pd.melt(df)

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
    if (y[0] == 0.9999999999999999 and y[1] == 0.9999999999999999) or (y[0] == 0.0 and y[1] == 0.0):
        continue
    else:
        ax.plot(x, y, color='salmon', alpha=0.2)
        y_all = np.vstack((y_all, y))

ave_slope = np.mean(y_all[1:, 1] - y_all[1:, 0])
plt.xlabel('')
plt.ylabel('clustering coeff.')
plt.plot([0, 1], [0.5 - ave_slope / 2, 0.5 + ave_slope / 2], 'k')
plt.suptitle(f'{condition} Context A')
plt.savefig(os.path.join(os.getcwd(), f"visualization/20210603/{condition}_clustering_conA_matched_remove_extreme.png"), transparent=True, dpi=300)
plt.show()

print(scipy.stats.ttest_rel(set1, set2))

# %% Context B data
set1 = cc_all_mice_1_B
set2 = cc_all_mice_9_B

# Put into dataframe
df = pd.DataFrame({'D1': set1, 'D9': set2})
data = pd.melt(df)

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
    if (y[0] == 0.9999999999999999 and y[1] == 0.9999999999999999) or (y[0] == 0.0 and y[1] == 0.0):
        continue
    else:
        ax.plot(x, y, color='turquoise', alpha=0.2)
        y_all = np.vstack((y_all, y))

ave_slope = np.mean(y_all[1:, 1] - y_all[1:, 0])
plt.xlabel('')
plt.ylabel('clustering coeff.')
plt.plot([0, 1], [0.5 - ave_slope / 2, 0.5 + ave_slope / 2], 'k')
plt.suptitle(f'{condition} Context B')
plt.savefig(os.path.join(os.getcwd(), f"visualization/20210603/{condition}_clustering_conB_matched_remove_extreme.png"), transparent=True, dpi=300)
plt.show()

print(scipy.stats.ttest_rel(set1, set2))



