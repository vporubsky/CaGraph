# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 07:11:00 2020

@author: Veronica Porubsky

Title: getContextConnectivity

Description: Determine which subsets of neurons are most correlated with each 
other in the two contexts. Would be interesting to see how this compares to which 
neurons are most active in each context. 

Place a threshold, and return connected subnetworks.
"""
import networkx as nx
from neuronal_network_graph import neuronal_network_graph as nng
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

day1_treated = ['348-1_D1_all_calcium_traces.npy', '349-2_D1_all_calcium_traces.npy','386-2_D1_all_calcium_traces.npy','387-4_D1_all_calcium_traces.npy','396-1_D1_all_calcium_traces.npy','396-3_D1_all_calcium_traces.npy']
day9_treated = ['348-1_D9_all_calcium_traces.npy', '349-2_D9_all_calcium_traces.npy','386-2_D9_all_calcium_traces.npy','387-4_D9_all_calcium_traces.npy','396-1_D9_all_calcium_traces.npy','396-3_D9_all_calcium_traces.npy']

day1_untreated = ['1055-1_D1_all_calcium_traces.npy',  '1055-2_D1_all_calcium_traces.npy','1055-3_D1_all_calcium_traces.npy', '1055-4_D1_all_calcium_traces.npy', '14-0_D1_all_calcium_traces.npy']
day9_untreated = ['1055-1_D9_all_calcium_traces.npy',  '1055-2_D9_all_calcium_traces.npy','1055-3_D9_all_calcium_traces.npy', '1055-4_D9_all_calcium_traces.npy', '14-0_D9_all_calcium_traces.npy']

all_files = [day1_treated, day9_treated, day1_untreated, day9_untreated]

#%% All measurements, separating contexts 
threshold = 0.5
names = []
data_mat = []

for treatment_group_index in range(0, 4):
    for mouse_id_index in range(len(all_files[treatment_group_index])):
        filename = all_files[treatment_group_index][mouse_id_index]
        mouse_id = filename.strip('_all_calcium_traces.npy')
        nn = nng(filename)
        num_neurons = nn.num_neurons
        
        con_act = np.load(mouse_id + '_neuron_context_active.npy')[0]
        
        con_A_active_indices = []
        con_B_active_indices = []
        for i in range(len(con_act)):
            if con_act[i] == 0:
                continue
            elif con_act[i] == 1:
                con_A_active_indices.append(i)
            elif con_act[i] == 2:
                con_B_active_indices.append(i)
        
        conA = nn.getContextAGraph(threshold = threshold)
        conB = nn.getContextBGraph(threshold = threshold)
        
        subnetwork_A = list(nx.connected_components(conA))
        subnetwork_B = list(nx.connected_components(conB))

        
        connected_subnetworks_A = []
        num_connected_subnetworks_A = 0
        len_connected_subnetworks_A = []
        
        connected_subnetworks_B = []
        num_connected_subnetworks_B = 0
        len_connected_subnetworks_B = []
        
        for k in range(len(subnetwork_A)):
            if len(subnetwork_A[k]) > 1:
                connected_subnetworks_A.append(list(map(int, subnetwork_A[k])))
                num_connected_subnetworks_A += 1
                len_connected_subnetworks_A.append(len(subnetwork_A[k]))
                
                
        for k in range(len(subnetwork_B)):
            if len(subnetwork_B[k]) > 1:
                connected_subnetworks_B.append(list(map(int, subnetwork_B[k]))) 
                num_connected_subnetworks_B += 1
                len_connected_subnetworks_B.append(len(subnetwork_B[k]))
        
        size_data_A = np.average(len_connected_subnetworks_A)
        size_data_B = np.average(len_connected_subnetworks_B)
        
        # compute percentage of neurons in subnetworks that are context-active neurons, regardless of complete
        active_in_connected_subnetwork_A = []
        for neuron in con_A_active_indices:
            for subnetwork in connected_subnetworks_A:
                if neuron in subnetwork:  
                    active_in_connected_subnetwork_A.append(neuron)
        num_context_A_active_neurons_in_context_A_connected_subnetworks = len(active_in_connected_subnetwork_A)
        
        active_in_connected_subnetwork_B = []
        for neuron in con_B_active_indices:
            for subnetwork in connected_subnetworks_B:
                if neuron in subnetwork:  
                    active_in_connected_subnetwork_B.append(neuron)
        num_context_B_active_neurons_in_context_B_connected_subnetworks = len(active_in_connected_subnetwork_B)

        # compute number of subnetworks entirely composed of context-active neurons
        num_fully_active_A = 0
        for subnetwork in connected_subnetworks_A:
            if all(elem in con_A_active_indices for elem in subnetwork):
                num_fully_active_A += 1
            
        num_fully_active_B = 0   
        for subnetwork in connected_subnetworks_B:
            if all(elem in con_B_active_indices for elem in subnetwork):
                num_fully_active_B += 1
       
        if treatment_group_index == 0 and mouse_id_index == 0:
            data = {mouse_id + '_A': [connected_subnetworks_A], mouse_id + '_B': [connected_subnetworks_B]}
            df = pd.DataFrame(data, index = ['connected subnetworks'])
            
            num_data = {mouse_id + '_A': [num_connected_subnetworks_A/num_neurons], mouse_id + '_B': [num_connected_subnetworks_B/num_neurons]}
            df2 = pd.DataFrame(num_data, index = ['num connected subnetworks'])
            
            size_data = {mouse_id + '_A': [size_data_A/num_neurons], mouse_id + '_B': [size_data_B/num_neurons]}
            df3 = pd.DataFrame(size_data, index = ['average size connected subnetworks'])
            
            percent_active_data ={mouse_id + '_A': num_context_A_active_neurons_in_context_A_connected_subnetworks/num_neurons, mouse_id + '_B': num_context_B_active_neurons_in_context_B_connected_subnetworks/num_neurons}
            df4 = pd.DataFrame(percent_active_data, index = ['num context-active neurons in connected subnetworks'])

            num_active_subnetworks = {mouse_id + '_A': num_fully_active_A/num_neurons, mouse_id + '_B': num_fully_active_B/num_neurons}
            df5 = pd.DataFrame(num_active_subnetworks, index = ['num active subnetworks'])
            
        else:
            df[mouse_id + '_A'] = [connected_subnetworks_A]   
            df[mouse_id + '_B'] = [connected_subnetworks_B]  
            
            df2[mouse_id + '_A'] = [num_connected_subnetworks_A/num_neurons]   
            df2[mouse_id + '_B'] = [num_connected_subnetworks_B/num_neurons]
            
            df3[mouse_id + '_A'] = [size_data_A/num_neurons]   
            df3[mouse_id + '_B'] = [size_data_B/num_neurons]
            
            df4[mouse_id + '_A'] = [num_context_A_active_neurons_in_context_A_connected_subnetworks/num_neurons]
            df4[mouse_id + '_B'] = [num_context_B_active_neurons_in_context_B_connected_subnetworks/num_neurons]

            df5[mouse_id + '_A'] = [num_fully_active_A/num_neurons]
            df5[mouse_id + '_B'] = [num_fully_active_B/num_neurons]
            
            
#%% Export data
# df.to_csv('connected_subnetworks_threshold_0_5.csv')
# df2.to_csv('connected_subnetworks_num_subnetworks_threshold_0_5.csv')
# df3.to_csv('connected_subnetworks_average_size_subnetworks_threshold_0_5.csv')
# df4.to_csv('connected_subnetworks_num_context_active_cells_in_subnetworks_threshold_0_5.csv')
# df5.to_csv('connected_subnetworks_num_composed_entirely_context_active_cells_threshold_0_5.csv')