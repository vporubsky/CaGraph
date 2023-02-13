"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 
File Final Edit Date:

Description: A script demonstrating each of the functions included in the neuronalNetworkGraph class
for future documentation generation.
"""
import numpy as np
from ca_graph import DGNetworkGraph as nng
import matplotlib.pyplot as plt
import networkx as nx
import os
from scipy import stats

#%% Set up neuronalNetworkGraph object and datasets
path = os.getcwd() + '/LC-DG-FC-data/'
data_file = '2-1_D1_smoothed_calcium_traces.csv'
nn = nng(data_file= path+data_file)
clustering_A = nn.get_context_A_clustering_coefficient()
clustering_B = nn.get_context_B_clustering_coefficient()

#%% Test functionality of plot_CDF()
plt.figure(figsize=(10,10))
nn.plot_CDF(data=clustering_A, color='salmon', x_label='Clustering Coefficient')
nn.plot_CDF(data=clustering_B, color='darkturquoise',x_label='Clustering Coefficient')
plt.show()

nn.plot_CDF_compare_two_samples(data_list=[clustering_A, clustering_B], color_list=['salmon', 'darkturquoise'], show_plot=True)