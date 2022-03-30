"""
Sample graphs for each behavior in the OFT or EZM, in the time period
corresponding to each stimulation applied.

This means that the timecourse was first split into stimulation period (pre-stim, stim, post-stim)
and then the behavior in that period was sampled to generate graphs in either the interior/exterior of the OFT
or the open/closed portion of the EZM.



"""
from neuronal_network_graph import DGNetworkGraph as nng
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl
import random

# Todo: This is from Sean's BLA data -- convert to DG

sns.set(style="white")
plt.rcParams.update({'font.size': 22})
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#a3cce2', '#7eb1d3',
                                                    '#5c96c5', '#3c7ab6',
                                                    '#205fa5', '#094492',
                                                    '#04287c', '#071971', '#0c0664'])

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


behavior = 'LC-DG-FC-data'
context = 'A'


def get_stim_indices(stim_selection):
    if stim_selection == 'A':
        return 0, 6000
    elif stim_selection == 'B':
        return 6000, 12000
    else:
        return 0, 18000

if behavior == 'EZM':
    path_to_data = "/Users/veronica_porubsky/GitHub/BLA_graph_theory/EZM/data/"
    location_labels = ['Open', 'Closed']
    behavior_label = '_closed.csv'
elif behavior == 'OFT':
    path_to_data = "/Users/veronica_porubsky/GitHub/BLA_graph_theory/OFT/data/"
    location_labels = ['Interior', 'Exterior']
    behavior_label = '_center_times.csv'


threshold = 0.3


# %%
if behavior == 'OFT':
    int_cc_all_mice = []
    ext_cc_all_mice = []

    for mouse_idx, data_file in enumerate(ca_data):

        mouse_id = data_file.replace('_deconTrace.csv', '')
        file = mouse_id + behavior_label
        data = np.genfromtxt(path_to_data + file, delimiter=",")
        start, end = get_stim_indices(stim_selection=stim_selection)
        data = np.genfromtxt(path_to_data + file, delimiter=",")[start:end]

        # find indices in binary array where animal is in the center
        if np.sum(data) >= 1:
            indices = [index for index, element in enumerate(data) if element == 1]
            shifted_ind = indices[1:]
            ind_new = indices[0:-1]
            # ones in the following array mean you have continuous timepoints that are in center
            # note: large numbers indicate a long time in exterior
            groups = np.array(shifted_ind) - np.array(ind_new)

            # find indices where there is a value > 15
            ext_ind = [index for index, element in enumerate(groups) if element > 15]
            exterior_indices = []
            for i in ext_ind:
                exterior_indices.append(indices[i])

            center_bound = []
            for i in range(len(exterior_indices)):
                center_bound.append(exterior_indices[i] + groups[ext_ind[i]])

            exterior_indices.append(indices[-1] + 1)

            # construct interior sub sample indices
            interior_indices = [(indices[0], exterior_indices[0])]
            for i in range(len(center_bound)):
                interior_indices.append((center_bound[i], exterior_indices[i + 1]))

            # construct exterior sub sample indices
            ext = [(0, indices[0] - 1)]
            for i in range(len(exterior_indices) - 1):
                ext.append((exterior_indices[i], center_bound[i] - 1))

            ext_int_len = [];
            for i in ext:
                ext_int_len.append(i[1] - i[0])

            int_int_len = [];
            for i in interior_indices:
                int_int_len.append(i[1] - i[0])

            print(f'{mouse_id} exterior bin sizes: {ext_int_len}')
            print(f'{mouse_id}: mean exterior bin sizes: {np.mean(ext_int_len)}')

            print(f'{mouse_id} interior bin sizes: {int_int_len}')
            print(f'{mouse_id}: mean interior bin sizes: {np.mean(int_int_len)}')

            # build distribution that looks like original distribution, then sample the size?
            new_ext_ind = []
            for i in ext:
                start_val = i[0]
                while start_val < i[1]:
                    cut_len = random.choice(int_int_len)
                    if (start_val + cut_len) < i[1]:
                        new_ext_ind.append((start_val, start_val + cut_len))
                    else:
                        new_ext_ind.append((start_val, i[1]))
                    start_val = start_val + cut_len
            ext = new_ext_ind


        else:
            interior_indices = [(start, end)]
            ext = None

elif behavior == 'EZM':
    # Todo: check open v. closed - currently first index is (0, -1)

    open_cluster_all_mice_avg = []
    closed_cluster_all_mice_avg = []

    for mouse_idx, data_file in enumerate(ca_data):

        closed_clustering = []
        open_clustering = []

        mouse_id = data_file.replace('_deconTrace.csv', '')
        file = mouse_id + behavior_label
        start, end = get_stim_indices(stim_selection=stim_selection)
        data = np.genfromtxt(path_to_data + file, delimiter=",")[start:end]

        # find indices in binary array where animal is in the closed portion of EZM
        indices = [index for index, element in enumerate(data) if
                   element == 1]  # if element = 1 --> in closed portion of maze
        shifted_ind = indices[1:]
        ind_new = indices[0:-1]
        # ones in the following array mean you have continuous timepoints that are in closed portion of the maze
        # note:  all other values (than 1) indicate that the mouse is in the open portion of the maze
        groups = np.array(shifted_ind) - np.array(ind_new)

        # find indices where there is a value > 1
        open_ind = [index for index, element in enumerate(groups) if element > 1]
        open_indices = []
        for i in open_ind:
            open_indices.append(indices[i])

        center_bound = []
        for i in range(len(open_indices)):
            center_bound.append(open_indices[i] + groups[open_ind[i]])

        open_indices.append(indices[-1] + 1)

        # construct closed sub sample indices
        closed = [(indices[0], open_indices[0])]
        for i in range(len(center_bound)):
            closed.append((center_bound[i], open_indices[i + 1]))

        # construct exterior sub sample indices
        open = [(0, indices[0] - 1)]
        for i in range(len(open_indices) - 1):
            open.append((open_indices[i], center_bound[i] - 1))

        open_ind_len = [];
        for i in open:
            open_ind_len.append(i[1] - i[0])

        closed_ind_len = [];
        for i in closed:
            closed_ind_len.append(i[1] - i[0])

        print(f'{mouse_id} open bin sizes: {open_ind_len}')
        print(f'{mouse_id}: mean open bin sizes: {np.mean(open_ind_len)}')

        print(f'{mouse_id} closed bin sizes: {closed_ind_len}')
        print(f'{mouse_id}: mean closed bin sizes: {np.mean(closed_ind_len)}')

        # build distribution that looks like original distribution, then sample the size?
        new_closed_ind = []
        for i in open:
            start_val = i[0]
            while start_val < i[1]:
                cut_len = random.choice(closed_ind_len)
                if (start_val + cut_len) < i[1]:
                    new_closed_ind.append((start_val, start_val + cut_len))
                else:
                    new_closed_ind.append((start_val, i[1]))
                start_val = start_val + cut_len

        closed = new_closed_ind
