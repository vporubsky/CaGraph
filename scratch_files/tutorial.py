"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 03-01-2023

Description: Short tutorial using ca_graph functionality.
"""
# Import packages
from cagraph import CaGraph
import visualization as viz
import preprocess as prep
import numpy as np
import os

# import cagraph.cagraph as cg
# import cagraph.visualization as viz
# import cagraph.preprocess as prep
# cg.CaGraph(data = ...)

DATA_PATH = os.getcwd() + '/datasets/'
# Todo: convert to Jupyter notebook for better organization/ flow

#%% Dataset and paths
# specify data file names, D1_WT contains a list of .csv files for the Day 1, WT condition
D1_WT = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv','1055-4_D1_smoothed_calcium_traces.csv',
         '122-1_D1_smoothed_calcium_traces.csv', '122-2_D1_smoothed_calcium_traces.csv', '122-3_D1_smoothed_calcium_traces.csv',
         '14-0_D1_smoothed_calcium_traces.csv']

# select file to use to generate graph
FILENAME = D1_WT[1]

#%% Set hyperparameters
THRESHOLD = 0.3

#%% Generate graph object, called "cg" from CSV file
# visualize CSV file in notebook
cg = CaGraph(DATA_PATH + FILENAME, dataset_id = '1055-1', threshold=THRESHOLD) # build CaGraph object
cg_graph = cg.get_network_graph() # Construct a graph

#%% Generate graph object from numpy.ndarray
data = np.genfromtxt(DATA_PATH + FILENAME, delimiter=',')
print(f"This dataset contains {data.shape[0] - 1} neurons and {data.shape[1]} timepoints.")

cg = CaGraph(data_file=data, dataset_id='1055-1', threshold=THRESHOLD)

#%% Dataset information
print(f"The dataset contains {cg.num_neurons} neurons and has a time interval of {cg.dt} sec")
print(f"Subject is {cg.data_id}")

#%% Generate graph object from NWB file (standardized format)
# Todo: add NWB file to datasets and add example

#%% Analyze graph topology
# compute the clustering coefficient for all nodes
cg_cc = cg.get_clustering_coefficient()

# compute the correlated pairs ratio for all nodes
cg_cr = cg.get_correlated_pair_ratio()

# compute the hubs in the graph
cg_hubs = cg.get_hubs()

#%% Standard graph visualization with NetworkX
cg.plot_graph_network(graph=cg_graph) # Plot the graph (simplistic version)

#%% interactive plotting with Bokeh integration
# generate interactive graph
viz.interactive_network(ca_graph_obj=cg,
                        adjust_size_by='degree',
                        adjust_color_by='communities')


#%% Change threshold and visualize graph
cg.threshold = 0.4
viz.interactive_network(ca_graph_obj=cg,
                        adjust_size_by='degree',
                        adjust_color_by='communities')

# Todo: Add demo for coloring by cell identifiers

#%% Plotting CDF to compare two conditions
# convert CSV to numpy.ndarry and index separate conditions
# (context A: second half of data)
cg_A = CaGraph(np.genfromtxt(DATA_PATH + FILENAME, delimiter=',')[:,1800:3600], threshold=THRESHOLD)
cg_A_cc = cg_A.get_clustering_coefficient()

# (context B: first half of data)
cg_B = CaGraph(np.genfromtxt(DATA_PATH+ FILENAME, delimiter=',')[:,0:1800], threshold=THRESHOLD)
cg_B_cc = cg_B.get_clustering_coefficient()

# plot histogram of distributions
viz.plot_histograms(data_list=[cg_A_cc, cg_B_cc],
                                x_label='cc',
                                color_list=['salmon','turquoise'],
                                bin_size=30,
                                show_plot=True)

# plot cumulative distribution function
viz.plot_CDF_compare_two_samples(data_list=[cg_A_cc, cg_B_cc],
                                             x_label='cc',
                                             color_list=['salmon', 'turquoise'],
                                             show_plot=True)

#%% Plotting matched samples
# Todo: check where additional figure canvas is appearing from
viz.plot_matched_data(sample_1=cg_A_cc,
                                  sample_2=cg_B_cc,
                                  labels=['A', 'B'],
                                  colors=['salmon','turquoise'],
                                  show_plot=True)

#%% Benchmarking with Preprocess class

# Generate shuffled dataset across individual neurons
data = np.genfromtxt(DATA_PATH + f'658-0_deconTrace.csv', delimiter=',')
event_data = np.genfromtxt(DATA_PATH + '658-0' + '_eventTrace.csv', delimiter=',')

# Shuffle the data using identified events
shuffled_data = prep.generate_event_shuffle(data=data.copy(), event_data=event_data)

#%% Plot shuffled trace
prep.plot_shuffle_example(data=data.copy(), shuffled_data=shuffled_data, event_data=event_data)


#%% Generate proposed threshold
threshold = prep.generate_threshold(data=data.copy(), shuffled_data=shuffled_data, event_data=event_data)

# plot threshold
prep.plot_threshold(data=data.copy(), shuffled_data=shuffled_data, event_data=event_data)


#%% Batched analyses
# Todo: add batched analyses (looping through datasets, generating averaged or aggregated results)
# Todo: add generate_report() --> provides whole-graph topology analysis using minimal inputs (first-pass understanding)
# Todo: add identifier information, binary behavior-binning example