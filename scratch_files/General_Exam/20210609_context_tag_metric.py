"""
Run correlated pairs ratio analysis and correlation coefficient analysis on
only the subset of cells that is preferentially active in context A.

Workflow:
1. Load matched context active data
2. Create unique identifiers for each context A active cell
3. Create networks as before
4. Run analyses as before
5. Pull out subset of data for context A active cells
"""
# Todo: need to run this analysis!
from neuronal_network_graph import DGNetworkGraph as nng
import numpy as np
import networkx as nx
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#%% Data files
# Wildtype condition
D1_WT = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv',
         '1055-4_D1_smoothed_calcium_traces.csv',
         '122-1_D1_smoothed_calcium_traces.csv',
         '122-2_D1_smoothed_calcium_traces.csv', '122-3_D1_smoothed_calcium_traces.csv', '14-0_D1_smoothed_calcium_traces.csv']
D9_WT = ['1055-1_D9_smoothed_calcium_traces.csv', '1055-2_D9_smoothed_calcium_traces.csv',
         '1055-4_D9_smoothed_calcium_traces.csv',
         '14-0_D9_smoothed_calcium_traces.csv', '122-1_D9_smoothed_calcium_traces.csv',
         '122-2_D9_smoothed_calcium_traces.csv', '122-3_D9_smoothed_calcium_traces.csv']

# Treatment condition
D1_Th = ['348-1_D1_smoothed_calcium_traces.csv',
         '349-2_D1_smoothed_calcium_traces.csv',
         '387-4_D1_smoothed_calcium_traces.csv', '396-1_D1_smoothed_calcium_traces.csv',
         '396-3_D1_smoothed_calcium_traces.csv']
D9_Th = ['387-4_D9_smoothed_calcium_traces.csv', '396-1_D9_smoothed_calcium_traces.csv',
         '396-3_D9_smoothed_calcium_traces.csv', '348-1_D9_smoothed_calcium_traces.csv','349-2_D9_smoothed_calcium_traces.csv']

# %% set hyper-parameters
threshold = 0.3