# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 00:39:37 2020

@author: Veronica Porubsky

Title: Test context A, B breakdown
"""
#%% teste context A and B separation
import networkx as nx
from dg_network_graph import DGNetworkGraph as nng
import matplotlib.pyplot as plt

path_to_data ='/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/LC-DG-FC-data/'
nn = nng(path_to_data + '1055-1_D1_smoothed_calcium_traces.csv')
threshold = 0.3
conA = nn.get_context_A_graph(threshold = threshold)
conB = nn.get_context_B_graph(threshold = threshold)

conA_null = nn.get_random_context_A_graph(threshold = threshold)
conB_null = nn.get_random_context_B_graph(threshold = threshold)

position = nx.spring_layout(conA)
plt.figure(1)
nx.draw_networkx_nodes(conA, pos = position, node_color = 'r', node_size=100)
nx.draw_networkx_edges(conA, pos = position, edge_color = 'r',)
nx.draw_networkx_labels(conA, pos = position, font_color = 'w', font_family='sans-serif')
plt.axis('off')
plt.show()

plt.figure(2)
nx.draw_networkx_nodes(conB, pos = position, node_color = 'b', node_size=100)
nx.draw_networkx_edges(conB, pos = position, edge_color = 'b',)
nx.draw_networkx_labels(conB, pos = position, font_color = 'w', font_family='sans-serif')
plt.axis('off')
plt.show()

plt.figure(3)
nx.draw_networkx_nodes(conA_null, pos = position, node_color = 'r', node_size=100)
nx.draw_networkx_edges(conA_null, pos = position, edge_color = 'r',)
nx.draw_networkx_labels(conA_null, pos = position, font_color = 'w', font_family='sans-serif')
plt.axis('off')
plt.show()

plt.figure(4)
nx.draw_networkx_nodes(conB_null, pos = position, node_color = 'b', node_size=100)
nx.draw_networkx_edges(conB_null, pos = position, edge_color = 'b',)
nx.draw_networkx_labels(conB_null, pos = position, font_color = 'w', font_family='sans-serif')
plt.axis('off')
plt.show()

