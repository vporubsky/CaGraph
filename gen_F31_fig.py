# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:01:06 2020

@author: Veronica Porubsky

Title: Generate F31 figures
"""
from neuronal_network_graph import neuronal_network_graph as nng
import numpy as np
import matplotlib.pyplot as plt
from mne.viz import plot_connectivity_circle


nn = nng('14-0_D1_all_calcium_traces.npy')
# nn_matched_D1 = nng(day1[index])
# nn_matched_D9 = nng(day9[index])
    
# graph_matched_D1_A = nn_matched_D1.get_context_A_graph(threshold = threshold)
# graph_matched_D9_A = nn_matched_D9.get_context_A_graph(threshold = threshold)
    
# graph_matched_D1_B = nn_matched_D1.get_context_B_graph(threshold = threshold)
# graph_matched_D9_B = nn_matched_D9.get_context_B_graph(threshold = threshold)
#%% 
fig_1 = plt.figure(1)       
nn.plot_correlation_heatmap(correlation_matrix = nn.get_pearsons_correlation_matrix(nn.neuron_dynamics[:10, 1800:3600]))
#plt.savefig('F31_correlation_mat_con_B.png', dpi = 300, transparent = True)


for i in range(0,10):
    y = nn.neuron_dynamics[i, :].copy()
    if i > 0: 
         for j in range(len(y)):
             y[j] = y[j] + 15*i
    plt.plot(nn.time, y, 'k', linewidth = 1) 
    plt.xticks([])
    plt.yticks([])
    
#plt.savefig('F31_timecourse.png', dpi = 300, transparent=True)

circle_net = nn.get_pearsons_correlation_matrix(nn.neuron_dynamics[:10, 0:1800])
fig_3 = plt.figure(3)
plot_connectivity_circle(con = circle_net, node_names = nn.labels[0:10], n_lines=20,\
                                 colormap = 'hot', node_colors = 'xkcd:grey', textcolor = 'xkcd:white',\
                                 colorbar = False, facecolor = 'xkcd:white')
#plt.savefig('F31_connectivity_A.png', dpi = 300, transparent = True)

circle_net = nn.get_pearsons_correlation_matrix(nn.neuron_dynamics[:10, 1800:3600])
fig_3 = plt.figure(4)
plot_connectivity_circle(con = circle_net, node_names = nn.labels[0:10], n_lines=20,\
                                 colormap = 'hot', node_colors = 'xkcd:grey', textcolor = 'xkcd:white',\
                                 colorbar = False, facecolor = 'xkcd:white')
#plt.savefig('F31_connectivity_B.png', dpi = 300, transparent = True)

#%%
circle_net = nn.get_pearsons_correlation_matrix(nn.neuron_dynamics[:10, 0:1800])
for i in range(len(circle_net)):
    for j in range(len(circle_net)):
        if circle_net[i, j] >= 0.1 and not i == j:
            circle_net[i, j] = 1
        else: 
            circle_net[i, j] = 0
plot_connectivity_circle(con = circle_net, node_names = nn.labels[:10], n_lines=8,\
                                 colormap = 'Spectral', textcolor = '#00000000',\
                                 colorbar = False, facecolor = '#00000000', node_edgecolor = 'w',  node_colors = 'k')


circle_net2 = nn.get_pearsons_correlation_matrix(nn.neuron_dynamics[:10, 1800:3600])
for i in range(len(circle_net2)):
    for j in range(len(circle_net2)):
        if circle_net2[i, j] >= 0.06 and not i == j:
            circle_net2[i, j] = 1
plot_connectivity_circle(con = circle_net2, node_names = nn.labels[:10], n_lines=8, node_edgecolor = 'w', node_colors = 'k', colormap = 'coolwarm', textcolor = '#00000000',colorbar = False, facecolor = '#00000000')
