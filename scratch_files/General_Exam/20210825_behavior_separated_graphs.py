"""
Run correlated pairs ratio analysis and correlation coefficient analysis on
graphs that are separated based on behavior.

Workflow:
1. Load
2. Create unique identifiers for each context A active cell
3. Create networks as before
4. Run analyses as before
5. Pull out subset of data for context A active cells
"""
# Import packages
from neuronal_network_graph import DGNetworkGraph as nng
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Todo: need to add catch for edge cases where behavior_A[i,1] == 360, some files only have time[-1] = 359.6
#%% Data files
# Wildtype condition
D1_WT = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv',
         '1055-4_D1_smoothed_calcium_traces.csv',
         '122-1_D1_smoothed_calcium_traces.csv',
         '122-2_D1_smoothed_calcium_traces.csv','122-3_D1_smoothed_calcium_traces.csv',
         '14-0_D1_smoothed_calcium_traces.csv']
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
         '396-3_D9_smoothed_calcium_traces.csv', '348-1_D9_smoothed_calcium_traces.csv',
         '349-2_D9_smoothed_calcium_traces.csv']

#%% Set filepath to freezing period datasets
filepath = os.getcwd() +'/freezing_periods/'

#%% Use freeze_data to partition graphs
# freeze_data will have 3 columns - start time frozen, end time frozen, and total time frozen in that period
# create a graph for the frozen periods and for the movement periods between the frozen periods
# need to add 180 to context A frozen time bins

threshold = 0.3
treatment = 'D9_WT'
treatment_group = D9_WT
cc_move_A = []
cc_freeze_A = []
cc_move_B = []
cc_freeze_B = []

for data_file in treatment_group:
    try:
        mouse_id = data_file.strip('_smoothed_calcium_traces.csv')
        behavior_file_A = np.genfromtxt(filepath + mouse_id + '_A.csv', delimiter=',')[1:, 1:4]
        behavior_file_B = np.genfromtxt(filepath + mouse_id + '_B.csv', delimiter=',')[1:, 1:4]

        # Instantiate neuronal_network_graph object
        nn = nng(data_file)
        nn_graph = nn.get_network_graph(threshold=threshold)
        print(f"Executing analyses for mouse: {mouse_id}")
        num_neurons = nn.num_neurons

        # Need time list to find indices using times in behavior times datasets
        time = list(map(float, list(nn.time)))

        # Process behavior times A dataset
        behavior_times_A = np.around(behavior_file_A, decimals=1)
        con_A_add = np.ones(np.shape(behavior_times_A))[:,0:2]*180
        behavior_times_A = behavior_times_A[:,0:2] + con_A_add

        start = np.array([i if int(i*10) % 2 == 0 else i + 0.1 for i in behavior_times_A[:,0]])
        end = np.array([i if int(i*10) % 2 == 0 else i + 0.1 for i in behavior_times_A[:,1]])
        behavior_A = np.transpose(np.vstack((start, end)))
        behavior_A = np.around(behavior_A, decimals=1)

        freeze_indices_A = []
        move_indices_A = []
        for i in range(np.shape(behavior_A)[0]):
            freeze_indices_A.append((time.index(behavior_A[i,0]), time.index(behavior_A[i,1])))
            if i < np.shape(behavior_A)[0] - 1:
                move_indices_A.append((time.index(behavior_A[i,1]), time.index(behavior_A[i + 1,0])))
        move_indices_A.append((time.index(behavior_A[i,1]), 1799)) # May need to add edge case here to get the end of timecourse


        move_graphs_A = nn.get_time_subsampled_graphs(subsample_indices=move_indices_A, threshold=threshold)
        freeze_graphs_A = nn.get_time_subsampled_graphs(subsample_indices=freeze_indices_A, threshold=threshold)


        # Process behavior times B dataset
        behavior_times_B = np.around(behavior_file_B, decimals=1)
        behavior_times_B = behavior_times_B[:,0:2]

        start = np.array([i if int(i*10) % 2 == 0 else i + 0.1 for i in behavior_times_B[:,0]])
        end = np.array([i if int(i*10) % 2 == 0 else i + 0.1 for i in behavior_times_B[:,1]])
        behavior_B = np.transpose(np.vstack((start, end)))
        behavior_B = np.around(behavior_B, decimals=1)

        freeze_indices_B = []
        move_indices_B = []
        for i in range(np.shape(behavior_B)[0]):
            freeze_indices_B.append((time.index(behavior_B[i,0]), time.index(behavior_B[i,1])))
            if i < np.shape(behavior_B)[0] - 1:
                move_indices_B.append((time.index(behavior_B[i,1]), time.index(behavior_B[i + 1,0])))
        move_indices_B.append((time.index(behavior_B[i,1]), 899))

        move_graphs_B = nn.get_time_subsampled_graphs(subsample_indices=move_indices_B, threshold=threshold)
        freeze_graphs_B = nn.get_time_subsampled_graphs(subsample_indices=freeze_indices_B, threshold=threshold)


        # Execute analyses
        cc_B_move = []
        cbm = []
        for idx, G in enumerate(move_graphs_B):
            # clustering coefficient
            clustering = nn.get_clustering_coefficient(graph=G, threshold=threshold)
            cc_B_move.append(clustering)
            cbm = cbm + clustering
        cc_move_B.append(np.median(clustering))

        # plt.figure(figsize=(15, 10))
        # plt.suptitle(mouse_id)
        # plt.subplot(1, 2, 1)

        # for idx in range(len(cc_B)):
        #     plt.hist(cc_B[idx], bins = 20, color='turquoise', alpha=0.5)
        #     plt.xlim((0, 1.0))
        #     plt.ylabel('CDF')
        #     plt.xlabel('clustering coefficient')
        #     plt.title('Move')

        cc_B_freeze = []
        cbf = []
        for idx, G in enumerate(freeze_graphs_B):
            # clustering coefficient
            clustering = nn.get_clustering_coefficient(graph=G, threshold=threshold)
            cc_B_freeze.append(clustering)
            cbf = cbf + clustering
        cc_freeze_B.append(np.median(clustering))

        # plt.subplot(1, 2, 2)
        # for idx in range(len(cc_B)):
        #     plt.xlabel('clustering coefficient')
        #     plt.title('Freeze')
        #     plt.hist(cc_B[idx], bins = 20, color='turquoise', alpha=0.5)
        #     plt.xlim((0, 1.0))

        # Execute analyses
        cc_A_move = []
        cam = []
        for idx, G in enumerate(move_graphs_A):
            # clustering coefficient
            clustering = nn.get_clustering_coefficient(graph=G, threshold=threshold)
            cc_A_move.append(clustering)
            cam = cam + clustering
        cc_move_A.append(np.median(clustering))

        # plt.subplot(1, 2, 1)
        #
        # for idx in range(len(cc_A)):
        #     plt.hist(cc_A[idx], bins = 20, color='mistyrose', alpha=0.5)
        #     plt.xlim((0, 1.0))

        cc_A_freeze = []
        caf = []
        for idx, G in enumerate(freeze_graphs_A):
            # clustering coefficient
            clustering = nn.get_clustering_coefficient(graph=G, threshold=threshold)
            cc_A_freeze.append(clustering)
            caf = caf + clustering
        cc_freeze_A.append(np.median(clustering))

        print(f'{mouse_id}, {treatment}: Context A, Freeze v Move')
        print(scipy.stats.ks_2samp(caf, cam))

        print(f'{mouse_id}, {treatment}: Context B, Freeze v Move')
        print(scipy.stats.ks_2samp(cbf, cbm))
    except:
        continue

    # plt.subplot(1, 2, 2)
    # for idx in range(len(cc_A)):
    #     plt.hist(cc_A[idx], bins = 20, color='mistyrose', alpha=0.5)
    #     plt.xlim((0, 1.0))
    # plt.savefig(os.getcwd() + '/network_visualization/' + mouse_id + '.png', dpi=300)
    # plt.show()


#%% Plot matched freeze-move
set1 = cc_move_B
set2 = cc_freeze_B

# Put into dataframe
df = pd.DataFrame({'Move': set1, 'Freeze': set2})
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
    ax.plot(locs1[i, 0], locs1[i, 1], '.', color='black')
    ax.plot(locs2[i, 0], locs2[i, 1], '.', color='black')
    data = [locs1[:, 1], locs2[:, 1]]
    plt.xticks([])
    y_all = np.vstack((y_all, y))

plt.ylabel('Median Clustering Coefficient')
plt.xlabel('Move                                          Freeze')
# plt.ylim((0.9,1.02))
# plt.yticks([0.9,1])
#plt.savefig(os.path.join(os.getcwd(), f"network_visualization/{mouse_id}_{treatment}_median.png"), transparent=True, dpi=300)
plt.show()

print(f'P-value = {scipy.stats.ttest_rel(set1, set2).pvalue:.3}')