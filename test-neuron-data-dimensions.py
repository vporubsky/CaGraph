# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:57:08 2020

@author: Veronica Porubsky


Title: test neuron dynamics dimension (ensure all neuron data is exported from matlab)
"""
from neuronalNetworkGraph import neuronalNetworkGraph
import numpy as np

nn = neuronalNetworkGraph('14-0_D1_all_calcium_traces.npy')

neuron_dynamics = nn.neuron_dynamics
print(np.shape(nn.neuron_dynamics))
print(len(nn.labels))