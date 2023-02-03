"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date:
File Final Edit Date:

Description: This file is used to only analyze the context-selective cells
"""
# Import packages
from setup import FC_DATA_PATH
from dg_network_graph import DGNetworkGraph as nng
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import os
from scipy import stats

path_to_data = FC_DATA_PATH
EXPORT_PATH = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/analyses/benchmarking/random_networks/scratch-analysis/'

treatment = ['2-1', '2-2', '2-3', '348-1', '349-2', '386-2', '387-4', '396-1']
WT = ['1055-1', '1055-2', '1055-3', '1055-4', '14-0', '122-1', '122-2', '122-3', '124-2']

#%% assemble list of ids
mouse_id_list = []
for file in os.listdir(path_to_data):
    if file.endswith('_smoothed_calcium_traces.csv'):
        print(file)
        mouse_id = file.replace('_smoothed_calcium_traces.csv', '')
        print(mouse_id)
        mouse_id_list.append(mouse_id)

#%% EVENT ANALYSIS: Only looking at Context B (first half of data)
# store: mouse_id, mean, median, max pearson, threshold event separated, ks-stat event separated, mean, median, max pearson, threshold bin-separated, ks-stat bin-separated
results = np.zeros((58, 13))
set_cs = 2 # 1 = context A, 0 = context B, -1 = nonselective

conB_event_num = []
plt.figure(figsize=(10,10))
for count, mouse in enumerate(mouse_id_list):
    file_str = mouse[:len(mouse)-3]
    if file_str in WT:
        group = 'WT'
        day = mouse[len(mouse)-2:]
        if day == 'D5' or day == 'D0':
            continue
        else:
            if day == 'D1':
                event_day = 'Day1'
            else:
                event_day = 'Day9'
            data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,0:1800]
            random_data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,:] # Use when you want to shuffle full timeseries
            event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')
            random_event_binned_data = generate_event_segmented(data=data.copy(), event_data=event_data)
            event_idx = list(np.nonzero(event_data[1, :])[0])
            len_event_idx = len(event_idx)
            for row in range(2, np.shape(event_data)[0]):
                event_idx = list(np.nonzero(event_data[row, :])[0])
                event_idx = [n for n in event_idx if n < 1800]
                len_event_idx += len(event_idx)
            conB_event_num.append(len_event_idx)
            print(f'There are: {len_event_idx} events')


# EVENT ANALYSIS: Only looking at Context A (second half of data)
# store: mouse_id, mean, median, max pearson, threshold event separated, ks-stat event separated, mean, median, max pearson, threshold bin-separated, ks-stat bin-separated
results = np.zeros((58, 13))

conA_event_num = []
plt.figure(figsize= (10,10))
for count, mouse in enumerate(mouse_id_list):
    file_str = mouse[:len(mouse)-3]
    if file_str in WT:
        group = 'WT'
        day = mouse[len(mouse)-2:]
        if day == 'D5' or day == 'D0':
            continue
        else:
            if day == 'D1':
                event_day = 'Day1'
            else:
                event_day = 'Day9'
            data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,0:1800]
            random_data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,:] # Use when you want to shuffle full timeseries
            event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')
            random_event_binned_data = generate_event_segmented(data=data.copy(), event_data=event_data)
            event_idx = list(np.nonzero(event_data[1, :])[0])
            len_event_idx = len(event_idx)
            for row in range(2, np.shape(event_data)[0]):
                event_idx = list(np.nonzero(event_data[row, :])[0])
                event_idx = [n for n in event_idx if n > 1800]
                len_event_idx += len(event_idx)
            conA_event_num.append(len_event_idx)
            print(f'There are: {len_event_idx} events')

#%%
import scipy.stats as stats
t_test_val = stats.ttest_rel(conA_event_num, conB_event_num)
print(f'WT mean con A: {np.mean(conA_event_num)}')
print(f'WT mean con B: {np.mean(conB_event_num)}')

# %% EVENT ANALYSIS: Only looking at Context B (first half of data)
# store: mouse_id, mean, median, max pearson, threshold event separated, ks-stat event separated, mean, median, max pearson, threshold bin-separated, ks-stat bin-separated
results = np.zeros((58, 13))
set_cs = 2  # 1 = context A, 0 = context B, -1 = nonselective

conB_event_num = []
plt.figure(figsize=(10, 10))
for count, mouse in enumerate(mouse_id_list):
    file_str = mouse[:len(mouse) - 3]
    if file_str in treatment:
        group = 'Th'
        day = mouse[len(mouse) - 2:]
        if day == 'D5' or day == 'D0':
            continue
        else:
            if day == 'D1':
                event_day = 'Day1'
            else:
                event_day = 'Day9'
            data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,
                   0:1800]
            random_data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[
                          :, :]  # Use when you want to shuffle full timeseries
            event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')
            random_event_binned_data = generate_event_segmented(data=data.copy(), event_data=event_data)
            event_idx = list(np.nonzero(event_data[1, :])[0])
            len_event_idx = len(event_idx)
            for row in range(2, np.shape(event_data)[0]):
                event_idx = list(np.nonzero(event_data[row, :])[0])
                event_idx = [n for n in event_idx if n < 1800]
                len_event_idx += len(event_idx)
            conB_event_num.append(len_event_idx)
            print(f'There are: {len_event_idx} events')

#  EVENT ANALYSIS: Only looking at Context A (second half of data)
# store: mouse_id, mean, median, max pearson, threshold event separated, ks-stat event separated, mean, median, max pearson, threshold bin-separated, ks-stat bin-separated
results = np.zeros((58, 13))

conA_event_num = []
plt.figure(figsize=(10, 10))
for count, mouse in enumerate(mouse_id_list):
    file_str = mouse[:len(mouse) - 3]
    if file_str in treatment:
        group = 'Th'
        day = mouse[len(mouse) - 2:]
        if day == 'D5' or day == 'D0':
            continue
        else:
            if day == 'D1':
                event_day = 'Day1'
            else:
                event_day = 'Day9'
            data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,
                   0:1800]
            random_data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[
                          :, :]  # Use when you want to shuffle full timeseries
            event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')
            random_event_binned_data = generate_event_segmented(data=data.copy(), event_data=event_data)
            event_idx = list(np.nonzero(event_data[1, :])[0])
            len_event_idx = len(event_idx)
            for row in range(2, np.shape(event_data)[0]):
                event_idx = list(np.nonzero(event_data[row, :])[0])
                event_idx = [n for n in event_idx if n > 1800]
                len_event_idx += len(event_idx)
            conA_event_num.append(len_event_idx)
            print(f'There are: {len_event_idx} events')

#
t_test_val = stats.ttest_rel(conA_event_num, conB_event_num)
print(f'Th mean con A: {np.mean(conA_event_num)}')
print(f'Th mean con B: {np.mean(conB_event_num)}')
print(f'Th: {t_test_val}')
