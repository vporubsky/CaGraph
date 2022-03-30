# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:17:49 2020

@author: vporu

Title: find retained high connections between contexts -- quantify using percentages
"""
import networkx as nx
from neuronalNetworkGraph import neuronalNetworkGraph
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

day1_treated = ['348-1_D1_all_calcium_traces.npy', '349-2_D1_all_calcium_traces.npy','386-2_D1_all_calcium_traces.npy','387-4_D1_all_calcium_traces.npy','396-1_D1_all_calcium_traces.npy','396-3_D1_all_calcium_traces.npy']
day9_treated = ['348-1_D9_all_calcium_traces.npy', '349-2_D9_all_calcium_traces.npy','386-2_D9_all_calcium_traces.npy','387-4_D9_all_calcium_traces.npy','396-1_D9_all_calcium_traces.npy','396-3_D9_all_calcium_traces.npy']

day1_untreated = ['1055-1_D1_all_calcium_traces.npy',  '1055-2_D1_all_calcium_traces.npy','1055-3_D1_all_calcium_traces.npy', '1055-4_D1_all_calcium_traces.npy', '14-0_D1_all_calcium_traces.npy']
day9_untreated = ['1055-1_D9_all_calcium_traces.npy',  '1055-2_D9_all_calcium_traces.npy','1055-3_D9_all_calcium_traces.npy', '1055-4_D9_all_calcium_traces.npy', '14-0_D9_all_calcium_traces.npy']

all_files = [day1_treated, day9_treated, day1_untreated, day9_untreated]

def commonMembers(edge_list1, edge_list2):
    common_member_list = []
    if (edge_list1 & edge_list2): 
        print(edge_list1 & edge_list2) 
        common_member_list.append(edge_list1 & edge_list2)
    else: 
        print("No common edges.")
        common_member_list.append('nan')
    return common_member_list

#%% All measurements
threshold = 0.5
names = []
data_mat = []

for treatment_group_index in range(0, 4):
    for mouse_id_index in range(len(all_files[treatment_group_index])):
        filename = all_files[treatment_group_index][mouse_id_index]
        mouse_id = filename.strip('_all_calcium_traces.npy')
        nn = neuronalNetworkGraph(filename)
        num_neurons = nn.num_neurons
        
        conA = nn.getContextAGraph(threshold = threshold)
        conA_edges = set(conA.edges)
        
        conB = nn.getContextBGraph(threshold = threshold)
        conB_edges = set(conB.edges)
        
        retained_correlation = commonMembers(conA_edges, conB_edges)   
        num_retained_correlations = len(retained_correlation[0])
                
        if treatment_group_index == 0 and mouse_id_index == 0:
            data = {mouse_id : retained_correlation}
            df = pd.DataFrame(data, index = ['edge present in context A and B'])
            
            num_data = {mouse_id : num_retained_correlations/num_neurons}
            df2 = pd.DataFrame(num_data, index = ['number common edges between context A, B'])
                      
        else:
            df[mouse_id] = retained_correlation
            df2[mouse_id] = num_retained_correlations/num_neurons
df.to_csv('common_edges_between_contexts_threshold_0_5.csv')
df2.to_csv('common_edges_between_contexts_number_threshold_0_5.csv')