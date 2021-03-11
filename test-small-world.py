# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:32:37 2020

@author: Veronica Porubsky

Title: test small world 
"""
import networkx as nx
print(nx.__version__)
from neuronal_network_graph import neuronal_network_graph as nng
import matplotlib.pyplot as plt

nn = nng('386-2_D1_all_calcium_traces.npy')
neuron_true, threshold, position = nn.get_network_graph(corr_matrix = nn.get_pearsons_correlation_matrix(nn.neuron_dynamics),threshold = 0.2)
neuron_null, position = nn.get_null_graph(corr_matrix = nn.get_pearsons_correlation_matrix(nn.neuron_dynamics), threshold = 0.2)
       
neuron_true_subgraph_generator = max(nx.connected_components(neuron_true), key=len)
neuron_true_subgraph = neuron_true.subgraph(neuron_true_subgraph_generator).copy() 
print(len(neuron_true_subgraph.nodes))
neuron_null_subgraph_generator = max(nx.connected_components(neuron_null), key=len)
neuron_null_subgraph = neuron_null.subgraph(neuron_null_subgraph_generator).copy() 
print(len(neuron_null_subgraph.nodes))

#%% Pure networkx implementation
true_stat = nx.algorithms.smallworld.sigma(neuron_true_subgraph)
null_stat = nx.algorithms.smallworld.sigma(neuron_null_subgraph)

plt.figure(1)
nx.draw_networkx_nodes(neuron_null_subgraph, pos = position, node_color = 'b', node_size=100)
nx.draw_networkx_edges(neuron_null_subgraph, pos = position, edge_color = 'b',)
nx.draw_networkx_labels(neuron_null_subgraph, pos = position, font_color = 'w', font_family='sans-serif')
plt.axis('off')
plt.show()


plt.figure(2)
nx.draw_networkx_nodes(neuron_true_subgraph, pos = position, node_color = 'b', node_size=100)
nx.draw_networkx_edges(neuron_true_subgraph, pos = position, edge_color = 'b',)
nx.draw_networkx_labels(neuron_true_subgraph, pos = position, font_color = 'w', font_family='sans-serif')
plt.axis('off')
plt.show()

#%% Using the function from neuronal_network_graph class
nn = nng('396-3_D1_all_calcium_traces.npy')
small_world_stat, small_world_subnetwork = nn.get_small_world_network_stat(corr_matrix = nn.get_pearsons_correlation_matrix(nn.neuron_dynamics), G= neuron_true, threshold = 0.2)
neuron_null, position = nn.get_null_graph(corr_matrix = nn.get_pearsons_correlation_matrix(nn.neuron_dynamics), threshold = 0.2)
print(len(small_world_subnetwork.nodes))  
small_world_stat2, small_world_subnetwork2 = nn.get_small_world_network_stat(corr_matrix = nn.get_pearsons_correlation_matrix(nn.neuron_dynamics), G = neuron_null, threshold = 0.2)
print(len(small_world_subnetwork2.nodes))
print(small_world_stat)
print(small_world_stat2)

#%%
nn = nng('396-3_D1_all_calcium_traces.npy')
small_world_stat, small_world_subnetwork = nn.get_small_world_network_stat(corr_matrix = nn.get_pearsons_correlation_matrix(nn.neuron_dynamics), G= neuron_true, threshold = 0.2)
neuron_null, position = nn.get_null_graph(corr_matrix = nn.get_pearsons_correlation_matrix(nn.neuron_dynamics), threshold = 0.2)
print(len(small_world_subnetwork.nodes))
#neuron_null_sw_stat, neuron_null_sw_subnetwork = nx.algorithms.smallworld.sigma(neuron_null)  
small_world_stat2, small_world_subnetwork2 = nn.get_small_world_network_stat(corr_matrix = nn.get_pearsons_correlation_matrix(nn.neuron_dynamics), G = neuron_null, threshold = 0.2)
print(len(small_world_subnetwork2.nodes))
print('small world stat is: {}'.format(small_world_stat))
print('small world stat null is: {}'.format(small_world_stat2))
#print('small world stat null is: {}'.format(neuron_null_sw_stat))

#%% test get small_world_network_stat_itr
from neuronal_network_graph import neuronal_network_graph as nng
nn = nng('396-3_D1_all_calcium_traces.npy')
neuron_true = nn.get_context_A_graph(threshold = 0.2)
sigma = nn.get_small_world_network_stat_itr(corr_matrix = nn.get_pearsons_correlation_matrix(nn.neuron_dynamics), G= neuron_true, threshold = 0.2)