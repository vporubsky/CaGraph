import networkx
from neuronal_network_graph import DGNetworkGraph as nng
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

subject_1 = ['1055-1_D1_smoothed_calcium_traces.csv','1055-1_D9_smoothed_calcium_traces.csv']

filename = subject_1[0]
mouse_id = filename.strip('_smoothed_calcium_traces.csv')
nn = nng(filename)

y = nn.neuron_dynamics.sum(axis=0)
tmp = stats.zscore(nn.neuron_dynamics, axis=1, nan_policy='omit')

plt.plot(np.linspace(0, len(tmp), len(tmp)), tmp)