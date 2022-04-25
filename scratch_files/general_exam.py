"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 04-24-2022
File Final Edit Date:

Description: 
"""

#%% Imports
import sys
import os
sys.path.insert(0, '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory')
from dg_network_graph import DGNetworkGraph as nng
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl

#%% Plotting and figure options
sns.set(style="white")
plt.rcParams.update({'font.size': 22})
export_path = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/scratch_files/General_Exam/'
dpi = 200

