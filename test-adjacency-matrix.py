# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:29:23 2020

@author: Veronica Porubsky

Title: test adjacency matrix 
"""
from neuronalNetworkGraph import neuronalNetworkGraph

nn = neuronalNetworkGraph('14-0_D1_all_calcium_traces.npy')

y = nn.getAdjacencyMatrix(corr_matrix = nn.getPearsonsCorrelationMatrix(nn.neuron_dynamics), threshold = 0.4)

