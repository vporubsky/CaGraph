import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os

#--------------------- TEMPORARY ------------------------------------#
# communicating with Sean Jewell to get updated FastLZeroSpikeInference Python package on Pip server
#setting temporary PATH variables
os.environ['R_HOME'] = 'C:\Program Files\R\R-3.5.0' #path to your R installation
os.environ['R_USER'] = 'C:\ProgramData\Anaconda3\Lib\site-packages\rpy2' #path depends on where you installed Python. Mine is the Anaconda distribution

# importing rpy2
import rpy2.robjects
from rpy2.robjects.packages import importr
lzsi = importr("LZeroSpikeInference")

#------------------- TEMPORARY --------------------------------------#
class comp_neuro_methods:
    """
    Created: 08/08/2029
    Author: Veronica Porubsky

    Class: comp_neuro_methods(npy_file)
    ======================
    Undergoing continued development.
    
    This class provides functionality to readily build computational neuroscience
    models informed by spike trains derived from experimental data.
    
    """
    
    def __init__(self, npy_file):
        self.data_filename = str(npy_file)
        self.data = np.load(npy_file)
        self.time = self.data[0, :]
    
    def get_poisson_spike_train(fr, dt, n_sec):
        return [int(i < fr*dt) for i in np.random.rand(1, n_sec*1000)[0]]

    def get_spike_times(spike_train):
        return [i for i, val in enumerate(spike_train) if val]
    
    def get_inter_spike_interval(spike_times):
        return [y - x - 1 for x,y in zip(spike_times, spike_times[1:])]