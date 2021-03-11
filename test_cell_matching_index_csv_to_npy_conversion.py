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
    if filename.endswith('_cellRegistered.npy'):
        os.remove(filename)
        
for filename in files:
    if filename.endswith('cellRegistered.csv'):
        mouse_id = filename.strip('_cellRegistered.csv')
        convertCSVToMatrix(filename, mouse_id + '_D1_D9_index_matching.npy')
        
#%% individual
filename = '14-0_cellRegistered.csv'
mouse_id = filename.strip('_cellRegistered.csv')
convertCSVToMatrix(filename, mouse_id + '_D1_D9_index_matching.npy')