# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 14:51:35 2020

@author: Veronica Porubsky

Title: Compare calcium fluorescence vs. spike train graphs

To do: break into context A and B
"""
from neuronal_network_graph import neuronal_network_graph as nng
import pickle
import numpy as np
import networkx as nx
import os 

def bin_spike_trains(spike_train_array, timepoints = None, bin_size = 5):
    if timepoints == 'con_A':
        spike_train_array = spike_train_array[:, 0:1800]
        end = 180
    elif timepoints == 'con_B':
        spike_train_array = spike_train_array[:, 1800:3600]
        end = 180
    else: 
        end = 360
    for bin_num in range(np.shape(spike_train_array)[0]):
        start_bin = 0
        end_bin = bin_size
        new_vector = []
        for j in range(0, end):
            new_vector.append(sum(spike_train_array[bin_num, start_bin:end_bin]))
            start_bin += bin_size
            end_bin += bin_size
        if bin_num == 0:
            binned_data = new_vector.copy()
        else:
            binned_data = np.vstack((binned_data, new_vector.copy()))
    return binned_data

def make_spike_array_graph(pearsons_binned_spikes, threshold):
    spike_graph = nx.Graph()
    labels = np.linspace(0, np.shape(pearsons_binned_spikes)[0]-1, np.shape(pearsons_binned_spikes)[0]).astype(int)
    for i in range(len(labels)):
        spike_graph.add_node(str(labels[i]))
        for j in range(len(labels)):
            if not i == j and pearsons_binned_spikes[i, j] > threshold: 
                spike_graph.add_edge(str(labels[i]), str(labels[j]), weight = pearsons_binned_spikes[i, j])
    return spike_graph

#%% check which edges are conserved 
files = os.listdir()
spike_threshold = 0.7
threshold = 0.7
bin_size = 10
tot_percent_conserved_spike_full = 0
tot_percent_conserved_ca_full = 0
tot_percent_conserved_spike_con_A = 0
tot_percent_conserved_ca_con_A = 0
tot_percent_conserved_spike_con_B = 0
tot_percent_conserved_ca_con_B = 0
num_mouse_id = 0

for file in files:
    if file.endswith('_spike_train.pkl'):
        mouse_id = file.replace('_spike_train.pkl', '')
        try:
            with open(file, 'rb') as f:
                data = pickle.load(f)
                
            binned_data_full = bin_spike_trains(data, bin_size = bin_size)
            binned_data_con_A = bin_spike_trains(data, timepoints = 'con_A', bin_size = bin_size)
            binned_data_con_B = bin_spike_trains(data, timepoints = 'con_B', bin_size = bin_size)
            binned_data_graph_full = make_spike_array_graph(np.corrcoef(binned_data_full), threshold = spike_threshold)
            binned_data_graph_con_A = make_spike_array_graph(np.corrcoef(binned_data_con_A), threshold = spike_threshold)
            binned_data_graph_con_B = make_spike_array_graph(np.corrcoef(binned_data_con_B), threshold = spike_threshold)
            
            nn = nng(mouse_id + '_all_calcium_traces.npy')
            ca_data_graph_full = nn.get_network_graph(nn.pearsons_correlation_matrix, threshold = threshold)[0]
            ca_data_graph_con_A = nn.get_context_A_graph(threshold = threshold)
            ca_data_graph_con_B = nn.get_context_B_graph(threshold = threshold)
            
            count_conserved_full = 0
            for i in range(len(list(binned_data_graph_full.edges))):
                if list(binned_data_graph_full.edges)[i] in list(ca_data_graph_full.edges):
                    #print("{} : the edge {} appears in spike train and calcium imaging correlations.".format(mouse_id, list(binned_data_graph.edges)[i]))
                    count_conserved_full +=1
            num_edge_spike_full = len(list(binned_data_graph_full.edges))
            num_node_spike_full = len(list(binned_data_graph_full.nodes))
            num_edge_ca_full = len(list(ca_data_graph_full.edges))
            num_node_ca_full = len(list(ca_data_graph_full.nodes))
            
            print("{} full: {}% spike conserved [{} edges, {} nodes]".format(mouse_id, round(count_conserved_full/num_edge_spike_full*100,2), num_edge_spike_full, num_node_spike_full))
            print("{} full: {}% calcium conserved [{} edges, {} nodes]".format(mouse_id, round(count_conserved_full/num_edge_ca_full*100, 2), num_edge_ca_full, num_node_ca_full))       
                    
            
            count_conserved_con_A = 0
            for i in range(len(list(binned_data_graph_con_A.edges))):
                if list(binned_data_graph_con_A.edges)[i] in list(ca_data_graph_con_A.edges):
                    #print("{} : the edge {} appears in spike train and calcium imaging correlations.".format(mouse_id, list(binned_data_graph.edges)[i]))
                    count_conserved_con_A +=1
            num_edge_spike_con_A = len(list(binned_data_graph_con_A.edges))
            num_node_spike_con_A = len(list(binned_data_graph_con_A.nodes))
            num_edge_ca_con_A = len(list(ca_data_graph_con_A.edges))
            num_node_ca_con_A = len(list(ca_data_graph_con_A.nodes))
            
            
            print("{} con A: {}% spike conserved [{} edges, {} nodes]".format(mouse_id, round(count_conserved_con_A/num_edge_spike_con_A*100,2), num_edge_spike_con_A, num_node_spike_con_A))
            print("{} con A: {}% calcium conserved [{} edges, {} nodes]".format(mouse_id, round(count_conserved_con_A/num_edge_ca_con_A*100, 2), num_edge_ca_con_A, num_node_ca_con_A))       
                    
            count_conserved_con_B = 0
            for i in range(len(list(binned_data_graph_con_B.edges))):
                if list(binned_data_graph_con_B.edges)[i] in list(ca_data_graph_con_B.edges):
                    #print("{} : the edge {} appears in spike train and calcium imaging correlations.".format(mouse_id, list(binned_data_graph.edges)[i]))
                    count_conserved_con_B +=1
            num_edge_spike_con_B = len(list(binned_data_graph_con_B.edges))
            num_node_spike_con_B = len(list(binned_data_graph_con_B.nodes))
            num_edge_ca_con_B = len(list(ca_data_graph_con_B.edges))
            num_node_ca_con_B = len(list(ca_data_graph_con_B.nodes))
            
            
            print("{} con B: {}% spike conserved [{} edges, {} nodes]".format(mouse_id, round(count_conserved_con_B/num_edge_spike_con_B*100,2), num_edge_spike_con_B, num_node_spike_con_B))
            print("{} con B: {}% calcium conserved [{} edges, {} nodes]".format(mouse_id, round(count_conserved_con_B/num_edge_ca_con_B*100, 2), num_edge_ca_con_B, num_node_ca_con_B))       
                    

            tot_percent_conserved_spike_full += round(count_conserved_full/num_edge_spike_full*100,2)
            tot_percent_conserved_ca_full += round(count_conserved_full/num_edge_ca_full*100, 2)
            
            tot_percent_conserved_spike_con_A += round(count_conserved_con_A/num_edge_spike_con_A*100,2)
            tot_percent_conserved_ca_con_A += round(count_conserved_con_A/num_edge_ca_con_A*100, 2)
            
            tot_percent_conserved_spike_con_B += round(count_conserved_con_B/num_edge_spike_con_B*100,2)
            tot_percent_conserved_ca_con_B += round(count_conserved_con_B/num_edge_ca_con_B*100, 2)
            num_mouse_id += 1

        except:
            continue
print(str(tot_percent_conserved_spike_full/num_mouse_id) + '% spike conserved in full') # aka % connections in spike correlation w
print(str(tot_percent_conserved_ca_full/num_mouse_id) + '% ca conserved in full')

print(str(tot_percent_conserved_spike_con_A/num_mouse_id) + '% spike conserved in con A')
print(str(tot_percent_conserved_ca_con_A/num_mouse_id) + '% ca conserved in con A')

print(str(tot_percent_conserved_spike_con_B/num_mouse_id) + '% spike conserved in con B')
print(str(tot_percent_conserved_ca_con_B/num_mouse_id) + '% ca conserved in con B')


#%% count number of spike in conB
    