# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 13:42:53 2020

@author: Veronica Porubsky

Title: Test random reference (to generate null model)
"""
import networkx as nx
from networkx.utils import not_implemented_for
from networkx.utils import py_random_state
from neuronalNetworkGraph import neuronalNetworkGraph
import matplotlib.pyplot as plt


# nx.random_reference(existing_graph_name, niter=1, connectivity=True, seed=None)

nn = neuronalNetworkGraph('mouse_1_with_treatment_day_9_all_calcium_traces.npy')
neuron_true, threshold, position = nn.instantiateNetworkGraph(threshold = 0.2)
neuron_null, threshold, position = nn.getNullGraph(threshold = 0.2)

plt.figure(1)
nn.plotGraphNetwork(G = neuron_null, threshold = threshold, position = position)
plt.figure(2)
nn.plotGraphNetwork(G = neuron_true, threshold = threshold, position = position)

plt.figure(3)
nn.plotCircleGraphNetwork(correlation_metric = 'Granger')

plt.figure(4)
nn.plotSingleNeuronTimeCourse(1)

plt.figure(5)
nn.plotAllNeuronsTimeCourse()



#%%% General plotting commands 
# G = nx.algorithms.smallworld.random_reference(neuron_true)

# plt.figure(1)
# nx.draw_networkx_nodes(neuron_null, pos = position, node_color = 'b', node_size=100)
# nx.draw_networkx_edges(neuron_null, pos = position, edge_color = 'b',)
# nx.draw_networkx_labels(neuron_null, pos = position, font_color = 'w', font_family='sans-serif')
# plt.axis('off')
# plt.show()


# plt.figure(2)
# nx.draw_networkx_nodes(G, pos = position, node_color = 'b', node_size=100)
# nx.draw_networkx_edges(G, pos = position, edge_color = 'b',)
# nx.draw_networkx_labels(G, pos = position, font_color = 'w', font_family='sans-serif')
# plt.axis('off')
# plt.show()