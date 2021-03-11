'''Run smallworld metric analysis separately due to computational time required to generate
randomized graphs.'''
from neuronal_network_graph import neuronal_network_graph as nng
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
sns.set(style="whitegrid")
my_pal = {"D1_A": "salmon", "D1_B": "darkturquoise", "D5_A": "salmon", "D5_B": "darkturquoise", "D9_A":"salmon", "D9_B":"darkturquoise"}
my_pal = {"rand_D1_A": "grey", "D1_A": "salmon", "rand_D1_B": "grey", "D1_B": "darkturquoise", "rand_D5_A": "grey", "D5_A": "salmon", "rand_D5_B": "grey","D5_B": "darkturquoise", "rand_D9_A": "grey", "D9_A":"salmon", "rand_D9_B": "grey", "D9_B":"darkturquoise"}

#%% Load untreated data files - WT
day1_untreated = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv', '1055-3_D1_smoothed_calcium_traces.csv', '1055-4_D1_smoothed_calcium_traces.csv', '14-0_D1_smoothed_calcium_traces.csv']
day5_untreated = ['1055-1_D5_smoothed_calcium_traces.csv', '1055-2_D5_smoothed_calcium_traces.csv','1055-3_D5_smoothed_calcium_traces.csv', '1055-4_D5_smoothed_calcium_traces.csv', '14-0_D5_smoothed_calcium_traces.csv']
day9_untreated = ['1055-1_D9_smoothed_calcium_traces.csv', '1055-2_D9_smoothed_calcium_traces.csv','1055-3_D9_smoothed_calcium_traces.csv', '1055-4_D9_smoothed_calcium_traces.csv', '14-0_D9_smoothed_calcium_traces.csv']

all_files = [day1_untreated, day5_untreated, day9_untreated]

#%% All measurements, separating contexts
threshold = 0.35
names = []
data_mat = []

# Store smallworld metrics
con_A_sw_D1 = []
con_B_sw_D1 = []
con_A_sw_D5 = []
con_B_sw_D5 = []
con_A_sw_D9 = []
con_B_sw_D9 = []


mouse_id_indices = []

# Loop through all subjects and perform smallworld analysis
for treatment_group_index in [0, 1, 2]:
    for mouse_id_index in range(len(all_files[treatment_group_index])):
        filename = all_files[treatment_group_index][mouse_id_index]
        mouse_id = filename.strip('_smoothed_calcium_traces.csv')

        if treatment_group_index == 0:
            mouse_id_indices.append(mouse_id.replace('_D1', ''))

        nn = nng(filename)
        print(f"Executing analyses for {mouse_id}")
        num_neurons = nn.num_neurons

        # Context A and B graphs
        conA = nn.get_context_A_graph(threshold=threshold)
        conB = nn.get_context_B_graph(threshold=threshold)

        # small-world analysis
        try:
            con_A_omega_val = nn.get_smallworld_largest_subnetwork(graph=conA, threshold=threshold)
            print(f'Completed analysis of {mouse_id} context A with omega val: {con_A_omega_val}')
        except:
            con_A_omega_val = float('nan')
            print(f'Completed analysis of {mouse_id} context A but omega val could not be computed.')
        try:
            con_B_omega_val = nn.get_smallworld_largest_subnetwork(graph=conB, threshold=threshold)
            print(f'Completed analysis of {mouse_id} context B with omega val: {con_B_omega_val}')
        except:
            con_B_omega_val = float('nan')
            print(f'Completed analysis of {mouse_id} context B but omega val could not be computed.')

        if treatment_group_index == 0:
            con_A_sw_D1.append(con_A_omega_val)
            con_B_sw_D1.append(con_B_omega_val)

        elif treatment_group_index == 1:
            con_A_sw_D5.append(con_A_omega_val)
            con_B_sw_D5.append(con_B_omega_val)

        elif treatment_group_index == 2:
            con_A_sw_D9.append(con_A_omega_val)
            con_B_sw_D9.append(con_B_omega_val)

# %% Compute small-world statistics
D1_A_B = scipy.stats.ttest_rel(con_A_sw_D1, con_B_sw_D1)
D9_A_B = scipy.stats.ttest_rel(con_A_sw_D9, con_B_sw_D9)
A_D1_D9 = scipy.stats.ttest_rel(con_A_sw_D1, con_A_sw_D9)
B_D1_D9 = scipy.stats.ttest_rel(con_B_sw_D1, con_B_sw_D9)

test_stats = ['D1 A and B: ' + str(D1_A_B.pvalue),'D9 A and B: ' + str(D9_A_B.pvalue),'A D1 and D9: ' + str(A_D1_D9.pvalue),'B D1 and D9: ' + str(B_D1_D9.pvalue)]
print('Small-world statistical test results:')
[print(x) for x in test_stats]
print('')

#%% plot smallworld metrics with seaborn
labels = ['D1_A', 'D1_B', 'D5_A', 'D5_B','D9_A','D9_B']
raw = [con_A_sw_D1,con_B_sw_D1,con_A_sw_D5,con_B_sw_D5,con_A_sw_D9,con_B_sw_D9]

plt.figure(figsize=(20,10))
data = pd.DataFrame(np.transpose(np.array(raw)), columns=labels, index=mouse_id_indices)
sns.swarmplot(data=data, color = 'k');
sns.boxplot(data=data, whis = 1.5, palette=my_pal);
plt.ylabel('Smallworld omega val')
plt.title('Edge weight threshold: ' + str(threshold))
plt.show()

data.to_csv(path_or_buf=f'smallworld_omega_vals_threshold_{threshold}.csv')

#%% To reload from saved dataframe and plot:
threshold = 0.35
data = pd.read_csv('smallworld_omega_vals_threshold_0.35.csv', index_col=0)
sns.swarmplot(data=data, color = 'k');
sns.boxplot(data=data, whis = 1.5, palette=my_pal);
plt.ylabel('Smallworld omega value')
plt.title('Edge weight threshold: ' + str(threshold))
plt.show()


#%% To reload from saved dataframe and plot:
threshold = 0.5
data = pd.read_csv('smallworld_omega_vals_threshold_0.5.csv', index_col=0)
data = data.dropna() # Todo: note that this drops mouse 14-0, which could not compute smallworld metric due to small subnetwork size
data = data.replace(-np.inf, -1) # Todo: check that approximation is appropriate
sns.swarmplot(data=data, color = 'k');
sns.boxplot(data=data, whis = 1.5, palette=my_pal);
plt.ylabel('Smallworld omega value')
plt.title('Edge weight threshold: ' + str(threshold))
plt.show()
