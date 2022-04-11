"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: April 11, 2022
File Final Edit Date:

Description: Initial analysis of event data.
"""

import numpy as np

# %% Global analysis parameters
threshold = 0.3
path_to_data = "/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/LC-DG-OFT-data/"

# %% Load untreated data files - saline

data_198_1 = ['198-1_Saline.csv', '198-1_Prop.csv', '198-1_Praz.csv', '198-1_Que5mgkg.csv', '198-1_CNO.csv',
              '198-1_CNOSaline.csv', '198-1_CNOProp.csv', '198-1_CNOPraz.csv', '198-1_CNOQue.csv']

data_202_4 = ['202-4_Saline.csv', '202-4_Prop.csv', '202-4_Praz.csv', '202-4_Que5mgkg.csv', '202-4_CNO.csv',
              '202-4_CNOSaline.csv', '202-4_CNOProp.csv', '202-4_CNOPraz.csv', '202-4_CNOQue.csv']

data_222_1 = ['222-1_Saline.csv', '222-1_Prop.csv', '222-1_Praz.csv', '222-1_Que5mgkg.csv', '222-1_CNO.csv',
              '222-1_CNOSaline.csv', '222-1_CNOProp.csv', '222-1_CNOPraz.csv', '222-1_CNOQue.csv']

data_223_3 = ['223-3_Saline.csv', '223-3_Prop.csv', '223-3_Praz.csv', '223-3_Que5mgkg.csv', '223-3_CNO.csv',
              '223-3_CNOSaline.csv', '223-3_CNOProp.csv', '223-3_CNOPraz.csv', '223-3_CNOQue.csv']

labels = ['Saline', 'Prop', 'Praz', 'Que 5mg/kg', 'CNO', 'CNO + Saline', 'CNO + Prop', 'CNO + Praz', 'CNO + Que']

#%%
def get_binned_event_traces(data_file, bin_size):
    data = np.genfromtxt(data_file, delimiter=",")
    np.shape(data)
    print(np.shape(data))
    print(bin_size)
    binned_data = np.zeros((np.shape(data)[0], int(np.shape(data)[1]/bin_size)))
    start_idx = 0
    for idx in range(np.shape(binned_data)[1]):
        print(idx)
        binned_data[:, idx] = data[:, start_idx: start_idx + bin_size].sum(axis=1)
        start_idx = start_idx + bin_size
    print(np.shape(binned_data))
    return binned_data


#%%
data_file = data_198_1[0]
mouse_id = data_file[0:5]
bin_size =
binned_data = get_binned_event_traces(data_file= path_to_data + data_file, bin_size=bin_size)