# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:56:46 2019

@author: Veronica Porubsky

Title: Test script for converting calcium imaging .csv data + plotting 
"""
from arrange_calcium_data import convertCSVToMatrix, getNumColsInCSV, getNumRowsInCSV
import matplotlib.pyplot as plt
import numpy as np

#%%
getNumColsInCSV('1055-3_D1_all_calcium_traces.csv')
getNumRowsInCSV('1055-3_D1_all_calcium_traces.csv')

#%% Convert data to numpy array
#matrix_data = convertCSVToMatrix('mouse_11_all_calcium_traces.csv', 'mouse_11_all_calcium_traces.npy')
matrix_data = convertCSVToMatrix('1055-3_D1_all_calcium_traces.csv', '1055-3_D1_all_calcium_traces.npy')
matrix_data = convertCSVToMatrix('1055-3_D9_all_calcium_traces.csv', '1055-3_D9_all_calcium_traces.npy')

#%% Plot timecourse data
plt.figure(figsize = (10, 3))
plt.xlim((0, 360))
plt.xticks([0, 90, 180, 270, 360])
for i in range(1, np.shape(matrix_data)[0]):
    plt.plot(matrix_data[0, :], matrix_data[i, :], linewidth=0.5)
    
#%%
from arrange_calcium_data import convertCSVToMatrix
import matplotlib.pyplot as plt
import numpy as np
from neuronalNetworkGraph import neuronalNetworkGraph
import os

files = os.listdir()
        
for file in files:
    if file.endswith('_all_calcium_traces.csv'):
        filename = file
        print(filename)
        mouse_id = filename.strip('csv')
        print(mouse_id)
        convertCSVToMatrix(filename, mouse_id + 'npy')