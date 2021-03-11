# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 11:20:09 2020

@author: Veronica Porubsky

Title: test cell matching index conversion

"""
from neuronal_network_graph import neuronal_network_graph
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


day1 = ['1055-1_D1_all_calcium_traces.npy',  '1055-2_D1_all_calcium_traces.npy','1055-4_D1_all_calcium_traces.npy', '14-0_D1_all_calcium_traces.npy', '348-1_D1_all_calcium_traces.npy', '349-2_D1_all_calcium_traces.npy','387-4_D1_all_calcium_traces.npy','396-1_D1_all_calcium_traces.npy','396-3_D1_all_calcium_traces.npy']
day9 = ['1055-1_D9_all_calcium_traces.npy',  '1055-2_D9_all_calcium_traces.npy','1055-4_D9_all_calcium_traces.npy', '14-0_D9_all_calcium_traces.npy', '348-1_D9_all_calcium_traces.npy', '349-2_D9_all_calcium_traces.npy','387-4_D9_all_calcium_traces.npy','396-1_D9_all_calcium_traces.npy','396-3_D9_all_calcium_traces.npy']


#%% Not in active development - see get_cell_matched_D1D9_stats.py
# for i in range(len(day1)):
#     D1_filename = day1[i]
#     D9_filename = day9[i]
#     mouse_id = D1_filename.replace('_D1_all_calcium_traces.npy', '')
        
#     # Load networks + indices
#     nn_D1 = neuronal_network_graph(D1_filename)
#     D1_A = nn_D1.get_context_A_graph(threshold = 0.5)
#     D1_B = nn_D1.get_context_B_graph(threshold = 0.5)
    
#     nn_D9 = neuronal_network_graph(D9_filename)
#     D9_A = nn_D9.get_context_A_graph(threshold = 0.5)
#     D9_B = nn_D9.get_context_B_graph(threshold = 0.5)
    
#     # Remove neurons which do not appear on both Day 1 and Day 9
#     cell_matching_indices = np.load(mouse_id + '_D1_D9_index_matching.npy')
#     del_row = []
#     for row in range(len(cell_matching_indices)):
#         if 0 in cell_matching_indices[row]:
#             del_row.append(row)
#     matched_indices = np.delete(cell_matching_indices, del_row, 0)
    
#     # decrement the values of indices - Matlab to Python indexing
#     matched_indices = np.subtract(matched_indices, np.ones(np.shape(matched_indices)))
#     matched_indices = matched_indices.astype(int)       
        
#     # Run analyses
#     retained_correlation_con_A = []
#     retained_correlation_con_B = []
#     for j in range(len(matched_indices)):
#         if any([str(matched_indices[j, 0]) in x for x in list(D1_A.edges)]) and any([str(matched_indices[j, 1]) in x for x in list(D9_A.edges)]):
#            # if this condition is met, a neuron which is matched on Day 1 and Day 9 are correlated with each other above the threshold on both days in context A
#             retained_correlation_con_A.append(j)
#         if any([str(matched_indices[j, 0]) in x for x in list(D1_B.edges)]) and any([str(matched_indices[j, 1]) in x for x in list(D9_B.edges)]):
#            # if this condition is met, a neuron which is matched on Day 1 and Day 9 are correlated with each other above the threshold on both days in context A
#             retained_correlation_con_B.append(j)