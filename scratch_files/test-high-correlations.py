# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 12:19:32 2020

@author: vporu

Title: test highly-correlated connections
"""
import networkx as nx
from neuronalNetworkGraph import neuronalNetworkGraph
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

day1 = ['1055-1_D1_all_calcium_traces.npy',  '1055-2_D1_all_calcium_traces.npy','1055-3_D1_all_calcium_traces.npy', '1055-4_D1_all_calcium_traces.npy', '14-0_D1_all_calcium_traces.npy', '348-1_D1_all_calcium_traces.npy', '349-2_D1_all_calcium_traces.npy','386-2_D1_all_calcium_traces.npy','387-4_D1_all_calcium_traces.npy','396-1_D1_all_calcium_traces.npy','396-3_D1_all_calcium_traces.npy']
day9 = ['1055-1_D9_all_calcium_traces.npy',  '1055-2_D9_all_calcium_traces.npy','1055-3_D9_all_calcium_traces.npy', '1055-4_D9_all_calcium_traces.npy', '14-0_D9_all_calcium_traces.npy', '348-1_D9_all_calcium_traces.npy', '349-2_D9_all_calcium_traces.npy','386-2_D9_all_calcium_traces.npy','387-4_D9_all_calcium_traces.npy','396-1_D9_all_calcium_traces.npy','396-3_D9_all_calcium_traces.npy']


nn = neuronalNetworkGraph('1055-1_D1_all_calcium_traces.npy')
nn.plotSingleNeuronTimeCourse(1)
nn.plotSingleNeuronTimeCourse(38)
nn.plotSingleNeuronTimeCourse(35)