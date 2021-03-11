# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 00:39:37 2020

@author: Veronica Porubsky

Title: Test context A, B breakdown
"""
#%% teste context A and B separation
import networkx as nx
from neuronalNetworkGraph import neuronalNetworkGraph
import matplotlib.pyplot as plt

nn = neuronalNetworkGraph('1055-1_D1_all_calcium_traces.npy')
threshold = 0.3
conA= nn.getContextAGraph(threshold = threshold)
conB = nn.getContextBGraph(threshold = threshold)

conA_null, position3 = nn.getNullContextAGraph(threshold = threshold)
conB_null, position4 = nn.getNullContextBGraph(threshold = threshold)

plt.figure(1)
nx.draw_networkx_nodes(conA, pos = position1, node_color = 'r', node_size=100)
nx.draw_networkx_edges(conA, pos = position1, edge_color = 'r',)
nx.draw_networkx_labels(conA, pos = position1, font_color = 'w', font_family='sans-serif')
plt.axis('off')
plt.show()

plt.figure(2)
nx.draw_networkx_nodes(conB, pos = position2, node_color = 'b', node_size=100)
nx.draw_networkx_edges(conB, pos = position2, edge_color = 'b',)
nx.draw_networkx_labels(conB, pos = position2, font_color = 'w', font_family='sans-serif')
plt.axis('off')
plt.show()

plt.figure(3)
nx.draw_networkx_nodes(conA_null, pos = position1, node_color = 'r', node_size=100)
nx.draw_networkx_edges(conA_null, pos = position1, edge_color = 'r',)
nx.draw_networkx_labels(conA_null, pos = position1, font_color = 'w', font_family='sans-serif')
plt.axis('off')
plt.show()

plt.figure(4)
nx.draw_networkx_nodes(conB_null, pos = position2, node_color = 'b', node_size=100)
nx.draw_networkx_edges(conB_null, pos = position2, edge_color = 'b',)
nx.draw_networkx_labels(conB_null, pos = position2, font_color = 'w', font_family='sans-serif')
plt.axis('off')
plt.show()

