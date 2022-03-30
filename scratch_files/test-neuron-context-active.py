# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:57:12 2020

@author: Veronica Porubsky
"""
import matplotlib.pyplot as plt
import numpy as np
from neuronalNetworkGraph import neuronalNetworkGraph
import networkx as nx
import os

con_act = np.load('387-4_D1_neuron_context_active.npy')[0] # load specifications for context neuron is active
nn = neuronalNetworkGraph('387-4_D1_all_calcium_traces.npy') # build neuronalNetworkGraph object

#%% Generate indices in the neuronalNetworkGraph that are context A and context B

con_A_active_indices = []
con_B_active_indices = []
for i in range(len(con_act)):
    if con_act[i] == 0:
        continue
    elif con_act[i] == 1:
        con_A_active_indices.append(i)
    elif con_act[i] == 2:
        con_B_active_indices.append(i)

#%%  from get_context_connectivity.py
threshold = 0.5
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

#%% check correspondence
print(connected_subnetworks_A)
print(con_A_active_indices)
print(connected_subnetworks_B)
print(con_B_active_indices)

#%% compute number of subnetworks entirely composed of context-active neurons
num_fully_active_A = 0
for subnetwork in connected_subnetworks_A:
    if all(elem in con_A_active_indices for elem in subnetwork):
        num_fully_active_A += 1
    
num_fully_active_B = 0   
for subnetwork in connected_subnetworks_B:
    if all(elem in con_B_active_indices for elem in subnetwork):
        num_fully_active_B += 1
            
# compute percentage of neurons in subnetworks that are context-active neurons, regardless of complete
active_in_connected_subnetwork_A = []
for neuron in con_A_active_indices:
    for subnetwork in connected_subnetworks_A:
        if neuron in subnetwork:  
            active_in_connected_subnetwork_A.append(neuron)
percent_context_A_active_neurons_in_context_A_connected_subnetworks = len(active_in_connected_subnetwork_A)/sum([len(listElem) for listElem in connected_subnetworks_A])
#note: using percentage could be an issue if there are no connected subnetworks - divide by zero

active_in_connected_subnetwork_B = []
for neuron in con_B_active_indices:
    for subnetwork in connected_subnetworks_B:
        if neuron in subnetwork:  
            active_in_connected_subnetwork_B.append(neuron)
percent_context_B_active_neurons_in_context_B_connected_subnetworks = len(active_in_connected_subnetwork_B)/sum([len(listElem) for listElem in connected_subnetworks_B])

#%% plot some examples
nn.plotSingleNeuronTimeCourse(0)
nn.plotSingleNeuronTimeCourse(16)
nn.plotSingleNeuronTimeCourse(25)
nn.plotSingleNeuronTimeCourse(4)

#%% test hub metrics