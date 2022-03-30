# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:40:51 2019

@author: Veronica Porubsky

Title: make subplots with connectivity circle
"""
from neuronalNetworkGraph import neuronalNetworkGraph
import matplotlib.pyplot as plt

#%% Compute stability
mouse_1_day_1 = neuronalNetworkGraph('mouse_1_with_treatment_day_1_all_calcium_traces.npy')
mouse_1_day_9 = neuronalNetworkGraph('mouse_1_with_treatment_day_9_all_calcium_traces.npy')

stability_day_1, indices_day_1 = mouse_1_day_1.getStability(num_folds = 20, time_context_switch = 1800, threshold = 0.7)
stability_day_9, indices_day_9 = mouse_1_day_9.getStability(num_folds = 20, time_context_switch = 1800, threshold = 0.7)

mouse_1_day_1.plotCircleGraphNetwork('Pearson')
#%%
plt.figure(figsize = (3,3))
mouse_1_day_1.plotCircleGraphNetworkStability(stability_day_1, indices = indices_day_1)
mouse_1_day_9.plotCircleGraphNetworkStability(stability_day_9, indices = indices_day_9)
