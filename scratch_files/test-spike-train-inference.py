# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 16:31:12 2019

@author: Veronica Porubsky

Title: Test Witten Lab Spike Train Inference
"""
import numpy as np
import matplotlib.pyplot as plt
from neuronal_network_graph import neuronal_network_graph as nng

# setting temporary PATH variables
import os
os.environ['R_HOME'] = 'C:\Program Files\R\R-3.5.0' #path to your R installation
os.environ['R_USER'] = 'C:\ProgramData\Anaconda3\Lib\site-packages\rpy2' #path depends on where you installed Python. Mine is the Anaconda distribution

# importing rpy2
import rpy2.robjects
from rpy2.robjects.packages import importr
lzsi = importr("LZeroSpikeInference")

#%% Use these commented lines to show that there are multiple libraries of R packages
#import rpy2.rinterface
#from rpy2.robjects.packages import importr
#base = importr('base')
#print(base._libPaths()) # ensure that the path here contains the LZeroSpikeInference directory with all files

#%% test using simulated data from the LZeroSpikeInference package
d = lzsi.simulateAR1(n = 500, gam = 0.998, poisMean = 0.009, sd = 0.15, seed = 8)
fit = lzsi.estimateSpikes(d[1], **{'gam':0.998, 'lambda':8, 'type':"ar1"})
spikes = np.array(fit[0])
fittedValues = np.array(fit[1])

plt.plot(np.linspace(0, 500, 500), fittedValues)
plt.eventplot(spikes, orientation='horizontal', linelengths=0.25, lineoffsets=-1, colors='k')
plt.plot(np.linspace(0, 500, 500),d[1], alpha=0.25)
plt.xticks([])
plt.yticks([])

#%% test on data from Bruchas Lab DG dataset
import pickle
test = nng('14-0_D1_all_calcium_traces.npy')
with open('14-0_D1_spike_train.pkl', 'rb') as f:
    data = pickle.load(f)
spike_times = []
for i in range(len(data[5,:])):
    if data[5,i] == 1:
        spike_times.append(i/10)
#%%
tmp = test.get_single_neuron_timecourse(5)
plt.figure(figsize = (10, 3))
# plt.plot(time, fittedValues)
plt.eventplot(spike_times, orientation='horizontal', linelengths=1, lineoffsets=-1, colors='k')
# plt.plot(time, neuron_calcium_data, alpha = 0.25)
plt.plot(tmp[0,:], tmp[1,:], alpha = 0.25)
plt.axvline(180, 0, 1, 'k')
plt.xlim(0, 360)
plt.yticks([])
plt.xticks([])
plt.savefig('tmp_spike_example.jpg', dpi = 300)

#%% test on subset of data from Bruchas Lab DG dataset
time = tmp[0, 500:1000].tolist()
neuron_calcium_data = tmp[1, 500:1000].tolist()

neuron_calcium_data = rpy2.robjects.vectors.FloatVector(neuron_calcium_data)
fit = lzsi.estimateSpikes(neuron_calcium_data, **{'gam':0.97, 'lambda':5, 'type':"ar1"})
spikes = np.array(fit[0])
spike_timepoints = []
for i in range(len(spikes)):
    spike_timepoints.append((time[int(spikes[i])]))
fittedValues = np.array(fit[1])

#
plt.figure(figsize = (10, 3))
plt.plot(time, fittedValues)
plt.eventplot(spike_timepoints, orientation='horizontal', linelengths=0.25, lineoffsets=-1, colors='k')
plt.plot(time, neuron_calcium_data, alpha = 0.25)
plt.xlim((time[0], time[-1]))
plt.yticks([])



