# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 06:52:38 2020

@author: Veronica Porubsky

Test return of spike traces as single array
"""
from neuronal_network_graph import neuronal_network_graph as nng
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

#%% iterative 
files = os.listdir()

for filename in files:
    if filename.endswith('_all_calcium_traces.npy'):
        new_filename = filename.replace('_all_calcium_traces.npy', '_spike_train.pkl')
        spike_times_filename = filename.replace('_all_calcium_traces.npy', '_spike_times.pkl')
        
        nn = nng(filename)
        spike_array, spikes = nn.infer_spike_array()
        
        f = open(new_filename, "wb")
        pickle.dump(spike_array, f)
        f.close()
        
        g = open(spike_times_filename, "wb")
        pickle.dump(spikes, g)
        g.close()


#%% test spike inference
# filename = '348-1_D9_all_calcium_traces.npy'
# new_filename = filename.replace('_all_calcium_traces.npy', '_spike_train.pkl')
# nn = nng(filename)
# spike_array, spikes = nn.infer_spike_array()
# f = open(new_filename, "wb")
# pickle.dump(spike_array, f)
# f.close()


#%%
# y = np.load('spike_times_1055-3_D9.npy', allow_pickle = True)
# plt.figure(4, figsize = (30, 20))
# # Draw a spike raster plot
# plt.eventplot(y, linelengths = 0.7)     
# # Provide the title for the spike raster plot
# plt.title('Mouse 1055-3, Day 9: spike raster plot')
# # Give x axis label for the spike raster plot
# plt.xlabel('Spike')
# # Give y axis label for the spike raster plot
# plt.ylabel('Neuron')
# # Display the spike raster plot
# plt.show()
# plt.xlim((0, 3600))
# plt.xticks([])
# plt.axvline(x=1800)
# plt.ylim((-0.5, np.shape(y)[0]))
# plt.savefig('1055-3_D9_inferred_spikes_eventplot_.png', dpi = 300)