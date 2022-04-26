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
plt.rcParams.update({'font.size': 16})
export_path = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/scratch_files/General_Exam/'
import_data_path = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/LC-DG-FC-data/'
dpi = 200

#%% Data
# Load treated data files - Th
D0_Th = ['348-1_D0_smoothed_calcium_traces.csv',
         '349-2_D0_smoothed_calcium_traces.csv', '386-2_D0_smoothed_calcium_traces.csv',
         '387-4_D0_smoothed_calcium_traces.csv', '396-1_D0_smoothed_calcium_traces.csv',
         '396-3_D0_smoothed_calcium_traces.csv']
D1_Th = ['348-1_D1_smoothed_calcium_traces.csv',
         '349-2_D1_smoothed_calcium_traces.csv', '386-2_D1_smoothed_calcium_traces.csv',
         '387-4_D1_smoothed_calcium_traces.csv', '396-1_D1_smoothed_calcium_traces.csv',
         '396-3_D1_smoothed_calcium_traces.csv']
D5_Th = ['348-1_D5_smoothed_calcium_traces.csv',
         '349-2_D5_smoothed_calcium_traces.csv', '386-2_D5_smoothed_calcium_traces.csv',
         '387-4_D5_smoothed_calcium_traces.csv', '396-1_D5_smoothed_calcium_traces.csv',
         '396-3_D5_smoothed_calcium_traces.csv']
D9_Th = ['348-1_D9_smoothed_calcium_traces.csv',
         '349-2_D9_smoothed_calcium_traces.csv', '386-2_D9_smoothed_calcium_traces.csv',
         '387-4_D9_smoothed_calcium_traces.csv', '396-1_D9_smoothed_calcium_traces.csv',
         '396-3_D9_smoothed_calcium_traces.csv']

all_Th_files = [D0_Th, D1_Th, D5_Th, D9_Th]

# Load untreated data files - WT
D0_WT = ['1055-1_D0_smoothed_calcium_traces.csv','1055-2_D0_smoothed_calcium_traces.csv',
         '1055-3_D0_smoothed_calcium_traces.csv','1055-4_D0_smoothed_calcium_traces.csv',
         '14-0_D0_smoothed_calcium_traces.csv']
D1_WT = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv',
         '1055-3_D1_smoothed_calcium_traces.csv', '1055-4_D1_smoothed_calcium_traces.csv',
         '14-0_D1_smoothed_calcium_traces.csv', '122-1_D1_smoothed_calcium_traces.csv',
         '122-2_D1_smoothed_calcium_traces.csv', '122-3_D1_smoothed_calcium_traces.csv']
D5_WT = ['1055-1_D5_smoothed_calcium_traces.csv', '1055-2_D5_smoothed_calcium_traces.csv',
         '1055-3_D5_smoothed_calcium_traces.csv', '1055-4_D5_smoothed_calcium_traces.csv',
         '14-0_D5_smoothed_calcium_traces.csv']
D9_WT = ['1055-1_D9_smoothed_calcium_traces.csv', '1055-2_D9_smoothed_calcium_traces.csv',
         '1055-3_D9_smoothed_calcium_traces.csv', '1055-4_D9_smoothed_calcium_traces.csv',
         '14-0_D9_smoothed_calcium_traces.csv', '122-1_D9_smoothed_calcium_traces.csv',
         '122-2_D9_smoothed_calcium_traces.csv', '122-3_D9_smoothed_calcium_traces.csv']

all_WT_files = [D0_WT, D1_WT, D5_WT, D9_WT]

#%% Analyses

#%% 3D PCA plotting test WT
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
    nn = nng(import_data_path + D1_data_list[idx], dataset_id = mouse_id)
    ndp = NeuronalDynamicsProjection(import_data_path + D1_data_list[idx])
    ax1, ax2 = ndp.plot_3D_projection(dataset=nn.context_A_dynamics[:, 0:1790], color='mistyrose', return_axes=True)
    ndp.plot_3D_projection(dataset=nn.context_B_dynamics[:, 0:1790], axs=[ax1,ax2], color='turquoise')

    nn = nng(import_data_path + D9_data_list[idx], dataset_id=mouse_id)
    ndp = NeuronalDynamicsProjection(import_data_path + D9_data_list[idx])
    ndp.plot_3D_projection(dataset=nn.context_A_dynamics[:, 0:1790], axs=[ax1,ax2], color='salmon')
    ndp.plot_3D_projection(dataset=nn.context_B_dynamics[:, 0:1790], axs=[ax1,ax2], color='teal')
    plt.suptitle(f'{nn.data_id}')
    plt.legend(['D1_A', 'D1_B', 'D9_A', 'D9_B'], loc='upper right')
    plt.savefig(export_path + f'WT_{mouse_id}_PCA_projection_d1_d9.png', dpi=dpi)
    plt.show()

#%% PCA Th

D1_data_list = ['348-1_D1_smoothed_calcium_traces.csv',
         '349-2_D1_smoothed_calcium_traces.csv', '386-2_D1_smoothed_calcium_traces.csv',
         '387-4_D1_smoothed_calcium_traces.csv', '396-1_D1_smoothed_calcium_traces.csv',
         '396-3_D1_smoothed_calcium_traces.csv']

D9_data_list = ['348-1_D9_smoothed_calcium_traces.csv',
         '349-2_D9_smoothed_calcium_traces.csv', '386-2_D9_smoothed_calcium_traces.csv',
         '387-4_D9_smoothed_calcium_traces.csv', '396-1_D9_smoothed_calcium_traces.csv',
         '396-3_D9_smoothed_calcium_traces.csv']

for idx, data_file in enumerate(D1_data_list):

    mouse_id = data_file[0:data_file.index('_D')]

    fig = plt.figure(figsize=[9, 4])
    nn = nng(import_data_path + D1_data_list[idx], dataset_id = mouse_id)
    ndp = NeuronalDynamicsProjection(import_data_path + D1_data_list[idx])
    ax1, ax2 = ndp.plot_3D_projection(dataset=nn.context_A_dynamics[:, 0:1790], color='mistyrose', return_axes=True)
    ndp.plot_3D_projection(dataset=nn.context_B_dynamics[:, 0:1790], axs=[ax1,ax2], color='turquoise')

    nn = nng(import_data_path + D9_data_list[idx], dataset_id=mouse_id)
    ndp = NeuronalDynamicsProjection(import_data_path + D9_data_list[idx])
    ndp.plot_3D_projection(dataset=nn.context_A_dynamics[:, 0:1790], axs=[ax1,ax2], color='salmon')
    ndp.plot_3D_projection(dataset=nn.context_B_dynamics[:, 0:1790], axs=[ax1,ax2], color='teal')
    plt.suptitle(f'{nn.data_id}')
    plt.legend(['D1_A', 'D1_B', 'D9_A', 'D9_B'], loc='upper right')
    plt.savefig(export_path + 'Th_pca' + f'Th_{mouse_id}_PCA_projection_d1_d9.png', dpi=dpi)
    plt.show()