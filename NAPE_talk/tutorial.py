"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 11-04-2022
File Final Edit Date: 11-04-2022

Description: Short tutorial of using CaGraph class (to be updated to CaGraph)
"""
# Import packages
# from setup import *
FC_DATA_PATH = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/LC-DG-FC-data/'
from ca_graph import CaGraph
from visualization import *
from benchmarking import *

#%% Dataset and paths
# Specify data file names, D1_WT contains a list of .csv files for the Day 1, WT condition
D1_WT = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv','1055-4_D1_smoothed_calcium_traces.csv',
         '122-1_D1_smoothed_calcium_traces.csv', '122-2_D1_smoothed_calcium_traces.csv', '122-3_D1_smoothed_calcium_traces.csv', '14-0_D1_smoothed_calcium_traces.csv']

# Select file to use to generate graph
FILENAME = '1055-1_D1_smoothed_calcium_traces.csv'
FILENAME = D1_WT[1]

#%% Set hyperparameters
THRESHOLD = 0.3

#%% Generate graph object, called "cg" from CSV file
# visualize CSV file in notebook
cg = CaGraph(FC_DATA_PATH + FILENAME, dataset_id = '1055-1', threshold=THRESHOLD) # build CaGraph object
cg_graph_con_A = cg.get_network_graph(threshold=THRESHOLD) # Construct a graph

#%% Generate graph object from NWB file


#%% Analyze graph topology
# compute the clustering coefficient for all nodes
cg_D1_A_cc = cg.get_context_clustering_coefficient(threshold=THRESHOLD)
cg_D1_B_cc = cg.get_context_clustering_coefficient(threshold=THRESHOLD)

# compute the correlated pairs ratio for all nodes
cg_D1_A_cr = cg.get_context_A_correlated_pair_ratio(threshold=THRESHOLD)
cg_D1_B_cr = cg.get_context_B_correlated_pair_ratio(threshold=THRESHOLD)

# compute the hubs in the graph
cg_D1_B_hubs = cg.get_hubs()

#%% Example plotting CDF to compare two conditions
cg.plot_CDF_compare_two_samples(data_list=[cg_D1_A_cc, cg_D1_B_cc], x_label='cc', color_list= ['salmon', 'turquoise'], show_plot=True)

#%% Standard graph visualization with NetworkX
cg.plot_graph_network(graph=cg_graph_con_A) # Plot the graph (simplistic version)

#%% Interactive plotting with Bokeh integration
interactive_network(ca_graph_obj=cg, adjust_size_by='degree') # Generate interactive graph