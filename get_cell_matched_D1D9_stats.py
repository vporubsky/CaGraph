# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:13:45 2020

@author: Veronica Porubsky

Title: Get cell-matched D1D9 stat - 
"""
from neuronal_network_graph import neuronal_network_graph as nng
import numpy as np
import matplotlib.pyplot as plt
import pickle

#%%
index = 1
threshold = 0.25

day1 = ['1055-1_D1_matched_traces.npy',  '1055-2_D1_matched_traces.npy','1055-4_D1_matched_traces.npy', '348-1_D1_matched_traces.npy', '349-2_D1_matched_traces.npy','387-4_D1_matched_traces.npy','396-1_D1_matched_traces.npy','396-3_D1_matched_traces.npy']
day9 = ['1055-1_D9_matched_traces.npy',  '1055-2_D9_matched_traces.npy','1055-4_D9_matched_traces.npy', '348-1_D9_matched_traces.npy', '349-2_D9_matched_traces.npy','387-4_D9_matched_traces.npy','396-1_D9_matched_traces.npy','396-3_D9_matched_traces.npy']

for i in range(len(day1)):
    index = i

    print("Analysis of matched neurons of mouse {} with threshold {}".format(day1[index].replace('_D1_matched_traces.npy', ""), threshold))
    nn_matched_D1 = nng(day1[index])
    nn_matched_D9 = nng(day9[index])
    
    
    graph_matched_D1_A = nn_matched_D1.get_context_A_graph(threshold = threshold)
    graph_matched_D9_A = nn_matched_D9.get_context_A_graph(threshold = threshold)
    
    graph_matched_D1_B = nn_matched_D1.get_context_B_graph(threshold = threshold)
    graph_matched_D9_B = nn_matched_D9.get_context_B_graph(threshold = threshold)
    
    
    for i in range(len(list(graph_matched_D1_A.edges))):
        if list(graph_matched_D1_A.edges)[i] in list(graph_matched_D9_A.edges):
            print("In context A, the edge {} is conserved from D1 to D9".format(list(graph_matched_D1_A.edges)[i]))
    for i in range(len(list(graph_matched_D1_B.edges))):
        if list(graph_matched_D1_B.edges)[i] in list(graph_matched_D9_B.edges):
            print("In context B, the edge {} is conserved from D1 to D9".format(list(graph_matched_D1_B.edges)[i]))
            
    # plt.figure(1)        
    # nn_matched_D9.plot_correlation_heatmap(correlation_matrix = nn_matched_D9.get_pearsons_correlation_matrix(nn_matched_D9.get_context_A_dynamics()))
    # plt.figure(2) 
    # nn_matched_D1.plot_correlation_heatmap(correlation_matrix = nn_matched_D1.get_pearsons_correlation_matrix(nn_matched_D1.get_context_A_dynamics()))
    # plt.figure(3) 
    # nn_matched_D9.plot_correlation_heatmap(correlation_matrix = nn_matched_D9.get_pearsons_correlation_matrix(nn_matched_D9.get_context_B_dynamics()))
    # plt.figure(4) 
    # nn_matched_D1.plot_correlation_heatmap(correlation_matrix = nn_matched_D1.get_pearsons_correlation_matrix(nn_matched_D1.get_context_B_dynamics()))
    
    
    print("The number of edges with a {} threshold in D1_A is: {}".format(threshold, len(graph_matched_D1_A.edges)))
    print("The number of edges with a {} threshold in D9_A is: {}".format(threshold, len(graph_matched_D9_A.edges)))
    print("The number of edges with a {} threshold in D1_B is: {}".format(threshold, len(graph_matched_D1_B.edges)))
    print("The number of edges with a {} threshold in D9_B is: {}".format(threshold, len(graph_matched_D9_B.edges)))