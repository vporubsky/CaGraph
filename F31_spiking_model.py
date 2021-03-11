# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 18:22:22 2020

@author: Veronica Porubsky
"""
import numpy as np
import matplotlib.pyplot as plt
from neuronal_network_graph import neuronal_network_graph as nng
import networkx as nx

def get_poisson_spike_train(fr, dt, n_sec):
    return [int(i < fr*dt) for i in np.random.rand(1, n_sec*1000)[0]]

def get_spike_times(spike_train):
    return [i for i, val in enumerate(spike_train) if val]

def get_inter_spike_interval(spike_times):
    return [y - x - 1 for x,y in zip(spike_times, spike_times[1:])]

#%% 
T = 360           # duration of the trial in seconds
tau_ref = 1e-2     # refractory time scale (in seconds)
r0 = 1        # constant rate (in hertz)
N_spikes = T * r0  # average number of spikes per trial
isi_homog = -np.log(np.random.rand(N_spikes)) / r0  # random ISIs at maximum rate
def thin_isi(isi_homog, tau_r):
    """Given an homogeneous Poisson train defined by the sequence of
    inter-event intervals `isi_homog`, return the corresponding sequence
    when a a non-homogeneous  The first argument is a list or array of
    inter-spike intervals, assumed to correspond to a homogeneous Poisson
    train. The second argument is the time constant of the recovery
    process, in the same units as the ISIs.
    """
    sp_times_homog = np.cumsum(isi_homog)  # spike sequence at original rate
    sp_times = []
    sp_times.append(sp_times_homog[0])  # 1st spike
    last_spike = sp_times[-1]

    x = np.random.rand(N_spikes - 1)
    for i, t in enumerate(sp_times_homog[1:]):
        z = 1 - np.exp(-(t - last_spike) / tau_r)
        # Thinning
        if (x[i] < z):
            sp_times.append(sp_times_homog[i+1])
            last_spike = sp_times[-1]
        else:
            continue

    sp_times = np.array(sp_times)  # convert list to array
    isi = np.diff(sp_times)        # difference between successive spikes
    return isi

isi_ref = thin_isi(isi_homog, tau_ref)

# Plot ISI histograms
plt.figure(1)
n_bins =40
xmin, xmax = 0, 0.8 * isi_ref.max()
bins = np.linspace(xmin, xmax, n_bins, endpoint=True)
isi_ref = thin_isi(isi_homog, tau_ref)
plt.hist(isi_ref, bins, density=1, facecolor='red', alpha=0.4)

T = 360          # duration of the trial in seconds
tau_ref = 1e-2     # refractory time scale (in seconds)
r0 = 3       # constant rate (in hertz)
N_spikes = T * r0  # average number of spikes per trial
isi_homog = -np.log(np.random.rand(N_spikes)) / r0  # random ISIs at maximum rate
isi_ref = thin_isi(isi_homog, tau_ref)
plt.hist(isi_ref, bins, density=1, facecolor='#00CED1', alpha=0.4)
plt.xlabel("Interspike interval (s)")
plt.setp(plt.gca(), 'yticklabels', [])  # remove ticklabels on y axis
plt.xlim((0,4))
plt.savefig('sparse_spiking_probability_distrib.png', dpi = 300)

#%% test on data from Bruchas Lab DG dataset
import pickle
test = nng('14-0_D1_all_calcium_traces.npy')
with open('14-0_D1_spike_train.pkl', 'rb') as f:
    data = pickle.load(f)
   
spike_times_A = get_spike_times(data[5,:180])
spike_times_B = get_spike_times(data[5,180:360])
 
#%%
tmp = test.get_single_neuron_timecourse(5)
plt.figure(2, figsize = (10, 3))
# plt.plot(time, fittedValues)
plt.eventplot([x/10 for x in get_spike_times(data[5,:])], orientation='horizontal', linelengths=1, lineoffsets=-1, colors='k')
# plt.plot(time, neuron_calcium_data, alpha = 0.25)
plt.plot(tmp[0,:], tmp[1,:], 'k')
plt.xlim(0, 360)
plt.yticks([])
plt.xticks([])
plt.savefig('tmp_spike_example.jpg', dpi = 300)

#%%
plt.figure(3)
y = test.get_context_A_graph(threshold = 0.15)
nx.draw(y, node_size = 150, node_color = '#00CED1')
plt.savefig('con_A_network_model.png', dpi =300)
plt.figure(4)
x = test.get_context_B_graph(threshold = 0.15)
nx.draw(x, node_size = 150, node_color = 'salmon')
plt.savefig('con_B_network_model.png', dpi = 300)
