# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:12:34 2020

@author: Veronica Porubsky

Title: test Pearson's correlation matrix parsing
"""
from neuronal_network_graph import neuronal_network_graph as nng
import numpy as np
import matplotlib.pyplot as plt
import pickle

def remove_weight(edge_list):
    edge_list_no_weight = []
    for i in range(len(edge_list)):
        edge_list_no_weight.append(edge_list[i][:2])
    return edge_list_no_weight

index = 0
threshold = 0.25

day1 = ['1055-1_D1_matched_traces.npy',  '1055-2_D1_matched_traces.npy','1055-4_D1_matched_traces.npy', '348-1_D1_matched_traces.npy', '349-2_D1_matched_traces.npy','387-4_D1_matched_traces.npy','396-1_D1_matched_traces.npy','396-3_D1_matched_traces.npy']
day9 = ['1055-1_D9_matched_traces.npy',  '1055-2_D9_matched_traces.npy','1055-4_D9_matched_traces.npy', '348-1_D9_matched_traces.npy', '349-2_D9_matched_traces.npy','387-4_D9_matched_traces.npy','396-1_D9_matched_traces.npy','396-3_D9_matched_traces.npy']

for i in range(len(day1)):
    index = i
    mouse_id = day1[index].replace('_D1_matched_traces.npy', '')
    nn_matched_D1 = nng(day1[index])
    nn_matched_D9 = nng(day9[index])
    num_max_vals = round(0.5*nn_matched_D1.num_neurons)
    
    
    graph_matched_D1_A = nn_matched_D1.get_context_A_graph(threshold = threshold)
    graph_matched_D9_A = nn_matched_D9.get_context_A_graph(threshold = threshold)
    
    graph_matched_D1_B = nn_matched_D1.get_context_B_graph(threshold = threshold)
    graph_matched_D9_B = nn_matched_D9.get_context_B_graph(threshold = threshold)
    
    D1A_max_weights = remove_weight(sorted(graph_matched_D1_A.edges(data=True),key= lambda x: x[2]['weight'],reverse=True)[:num_max_vals])
    D9A_max_weights = remove_weight(sorted(graph_matched_D9_A.edges(data=True),key= lambda x: x[2]['weight'],reverse=True)[:num_max_vals])
    D1B_max_weights = remove_weight(sorted(graph_matched_D1_B.edges(data=True),key= lambda x: x[2]['weight'],reverse=True)[:num_max_vals])
    D9B_max_weights = remove_weight(sorted(graph_matched_D9_B.edges(data=True),key= lambda x: x[2]['weight'],reverse=True)[:num_max_vals])
    
    count_shared_edge_conA = 0
    for j in D1A_max_weights: 
        if j in D9A_max_weights: 
            count_shared_edge_conA += 1
    print('{}: {} shared edges in top {} strongest correlated timeseries in con A between D1 and D9'.format(mouse_id, count_shared_edge_conA, num_max_vals))
            
    
    count_shared_edge_conB = 0
    for j in D1B_max_weights: 
        if j in D9B_max_weights: 
            count_shared_edge_conB += 1
    print('{}: {} shared edges in top {} strongest correlated timeseries in con B between D1 and D9'.format(mouse_id, count_shared_edge_conB, num_max_vals))