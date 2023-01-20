"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 11-04-2022
File Final Edit Date: 11-04-2022

Description: Short tutorial of using NeuronalNetworkGraph class (to be updated to CaGraph)
"""
# Import packages
from setup import FC_DATA_PATH
from dg_network_graph import DGNetworkGraph as nng

#%% Dataset and paths
# Specify data file names, D1_WT contains a list of .csv files for the Day 1, WT condition
D1_WT = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv','1055-4_D1_smoothed_calcium_traces.csv',
         '122-1_D1_smoothed_calcium_traces.csv', '122-2_D1_smoothed_calcium_traces.csv', '122-3_D1_smoothed_calcium_traces.csv', '14-0_D1_smoothed_calcium_traces.csv']

# Select file to use to generate graph
FILENAME = '1055-1_D1_smoothed_calcium_traces.csv'

#%% Set hyperparameters
THRESHOLD = 0.3

#%% Generate graph object, called "nn"
nn = nng(FC_DATA_PATH + FILENAME, dataset_id = '1055-1', threshold=THRESHOLD) # build CaGraph object
nn_graph_con_A = nn.get_context_A_graph(threshold=THRESHOLD) # Construct a graph
nn.plot_graph_network(graph=nn_graph_con_A) # Plot the graph (simplistic version)

#%% Analyze graph topology
# compute the clustering coefficient for all nodes
nn_D1_A_cc = nn.get_context_A_clustering_coefficient(threshold=THRESHOLD)
nn_D1_B_cc = nn.get_context_B_clustering_coefficient(threshold=THRESHOLD)

# compute the correlated pairs ratio for all nodes
nn_D1_A_cr = nn.get_context_A_correlated_pair_ratio(threshold=THRESHOLD)
nn_D1_B_cr = nn.get_context_B_correlated_pair_ratio(threshold=THRESHOLD)

#%% Example plotting CDF to compare two conditions
nn.plot_CDF_compare_two_samples(data_list=[nn_D1_A_cc, nn_D1_B_cc], x_label='cc', show_plot=True)