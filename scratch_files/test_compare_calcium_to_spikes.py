# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:51:35 2020

@author: Veronica Porubsky

Title: Compare calcium fluorescence vs. spike train graphs
"""
from neuronal_network_graph import neuronal_network_graph as nng

nn = nng('14-0_D1_all_calcium_traces.npy')
# spike_array, spike_times = nn.infer_spike_array()

#%% 
import numpy as np
pearsons_spikes = np.corrcoef(spike_array)

#%%
import networkx as nx
spike_graph = nx.Graph()
threshold = 0.5
labels = np.linspace(0, np.shape(nn.neuron_dynamics)[0]-1, np.shape(nn.neuron_dynamics)[0]).astype(int)
for i in range(len(labels)):
    spike_graph.add_node(str(labels[i]))
    for j in range(len(labels)):
                if not i == j and pearsons_binned_spikes[i, j] > threshold: 
                    spike_graph.add_edge(str(labels[i]), str(labels[j]), weight = pearsons_binned_spikes[i, j])      

nn_graph = nn.get_network_graph(corr_matrix = nn.pearsons_correlation_matrix, threshold = 0.5)

#%% Use pickle to save and load spike trains, too long to compute in real-time 
import pickle 
f = open("14-0_D1_spike_train.pkl","wb")
pickle.dump(spike_array,f)
f.close()

with open('14-0_D1_spike_train.pkl', 'rb') as f:
    data = pickle.load(f)

#%% Binned spike train data to compute correlations
for i in range(np.shape(data)[0]):
    start_bin = 0
    end_bin = 10
    new_vector = []
    for j in range(0, 360):
        new_vector.append(sum(data[i, start_bin:end_bin]))
        start_bin += 10
        end_bin += 10
    if i == 0:
        binned_data = new_vector.copy()
    else:
        binned_data = np.vstack((binned_data, new_vector.copy()))
        
    
    