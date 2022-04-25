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



#%% Analyses

#%% 3D PCA plotting test
from dg_network_graph import DGNetworkGraph as nng
from neuronal_dynamics_projection import NeuronalDynamicsProjection

D1_data_list = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv',
                 '1055-4_D1_smoothed_calcium_traces.csv', '122-1_D1_smoothed_calcium_traces.csv',
                 '122-2_D1_smoothed_calcium_traces.csv', '122-3_D1_smoothed_calcium_traces.csv',
                 '14-0_D1_smoothed_calcium_traces.csv']

D9_data_list = ['1055-1_D9_smoothed_calcium_traces.csv', '1055-2_D9_smoothed_calcium_traces.csv',
                 '1055-4_D9_smoothed_calcium_traces.csv', '122-1_D9_smoothed_calcium_traces.csv',
                 '122-2_D9_smoothed_calcium_traces.csv', '122-3_D9_smoothed_calcium_traces.csv',
                 '14-0_D9_smoothed_calcium_traces.csv']

for idx, data_file in enumerate(D1_data_list):

    mouse_id = data_file[0:data_file.index('_D')]

    fig = plt.figure(figsize=[9, 4])
    nn = nng(os.getcwd() + '/LC-DG-FC-data/' + D1_data_list[0], dataset_id = mouse_id)
    ndp = NeuronalDynamicsProjection(os.getcwd() + '/LC-DG-FC-data/' + D1_data_list[0])
    ax1, ax2 = ndp.plot_3D_projection(dataset=nn.context_A_dynamics[:, 0:1790], color='mistyrose', return_axes=True)
    ndp.plot_3D_projection(dataset=nn.context_B_dynamics[:, 0:1790], axs=[ax1,ax2], color='turquoise')

    nn = nng(os.getcwd() + '/LC-DG-FC-data/' + D9_data_list[idx], dataset_id=mouse_id)
    ndp = NeuronalDynamicsProjection(os.getcwd() + '/LC-DG-FC-data/' + D9_data_list[0])
    ndp.plot_3D_projection(dataset=nn.context_A_dynamics[:, 0:1790], axs=[ax1,ax2], color='salmon')
    ndp.plot_3D_projection(dataset=nn.context_B_dynamics[:, 0:1790], axs=[ax1,ax2], color='teal')
    plt.suptitle(f'{nn.data_id}')
    plt.legend(['D1_A', 'D1_B', 'D9_A', 'D9_B'], loc='upper right')
    plt.show()
