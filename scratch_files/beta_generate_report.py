"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 
File Final Edit Date:

Description: 
"""
# Import packages
from setup import FC_DATA_PATH
from dg_network_graph import DGNetworkGraph as nng
import numpy as np
import matplotlib.pyplot as plt
from generate_report import *
import os
path_to_data = FC_DATA_PATH
from scipy import stats
import pandas as pd

RANDOM_DATA_PATH = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/analyses/benchmarking/random_networks/randomized_neural_data/'
EXPORT_PATH = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/analyses/benchmarking/random_networks/scratch-analysis/'


#%% assemble list of ids
mouse_id_list = []
for file in os.listdir(path_to_data):
    if file.endswith('_smoothed_calcium_traces.csv'):
        print(file)
        mouse_id = file.replace('_smoothed_calcium_traces.csv', '')
        print(mouse_id)
        mouse_id_list.append(mouse_id)

#%% Only looking at Context B (first half of data)
# store: mouse_id, mean, median, max pearson, threshold event separated, ks-stat event separated, mean, median, max pearson, threshold bin-separated, ks-stat bin-separated
results = np.zeros((58, 13))

num_neurons = []
for count, mouse in enumerate(mouse_id_list):
    print(mouse)
    data = np.genfromtxt(path_to_data + f'/{mouse}_smoothed_calcium_traces.csv', delimiter=',')
    print(f'Num neurons: {data.shape[0]}')
    num_neurons += [data.shape[0]]

#%%
df = pd.DataFrame(num_neurons, mouse_id_list, columns=['Num neurons'])
mean_num = df['Num neurons'].mean()
max_num = df['Num neurons'].max()
min_num = df['Num neurons'].min()

#%% Loop through files and determine mean threshold for separation --> do this with event trace analysis and bin shuffle analysis