# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:57:12 2020

@author: Veronica Porubsky
"""

from arrange_calcium_data import convertCSVToMatrix
import matplotlib.pyplot as plt
import numpy as np
from neuronalNetworkGraph import neuronalNetworkGraph
import os

files = os.listdir()

for filename in files:
    if filename.endswith('neuron_context_active.npy'):
        os.remove(filename)
        
for filename in files:
    if filename.endswith('neuron_context_active.csv'):
        mouse_id = filename.strip('.csv')
        convertCSVToMatrix(filename, mouse_id + '.npy')
        
        #%% individual
filename = '1055-3_D1_neuron_context_active.csv'
mouse_id = filename.strip('.csv')
convertCSVToMatrix(filename, mouse_id + '.npy')