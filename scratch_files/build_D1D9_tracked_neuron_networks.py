# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:39:13 2020

@author: Veronica Porubsky

Title: Build networks composed only of neurons recorded + tracked on D1 and D9
"""
import numpy as np


day1 = ['1055-1_D1_all_calcium_traces.npy',  '1055-2_D1_all_calcium_traces.npy','1055-4_D1_all_calcium_traces.npy', '348-1_D1_all_calcium_traces.npy', '349-2_D1_all_calcium_traces.npy','387-4_D1_all_calcium_traces.npy','396-1_D1_all_calcium_traces.npy','396-3_D1_all_calcium_traces.npy']
day9 = ['1055-1_D9_all_calcium_traces.npy',  '1055-2_D9_all_calcium_traces.npy','1055-4_D9_all_calcium_traces.npy', '348-1_D9_all_calcium_traces.npy', '349-2_D9_all_calcium_traces.npy','387-4_D9_all_calcium_traces.npy','396-1_D9_all_calcium_traces.npy','396-3_D9_all_calcium_traces.npy']


#%%
np.genfromtxt(day1[0].replace('_D1_smoothed_calcium_traces.csv', 'cell_'), delimiter=",")

#%% 

# load file
for i in range(len(day1)):
    D1_file = np.load(day1[i])
    D9_file = np.load(day9[i])
    mouse_id = day1[i].replace('_D1_all_calcium_traces.npy', '')
    
    # Remove neurons which do not appear on both Day 1 and Day 9
    cell_matching_indices = np.load(mouse_id + '_D1_D9_index_matching.npy')
    del_row = []
    for row in range(len(cell_matching_indices)):
        if 0 in cell_matching_indices[row]:
            del_row.append(row)
    matched_indices = np.delete(cell_matching_indices, del_row, 0)

    # decrement the values of indices - Matlab to Python indexing
    matched_indices = np.subtract(matched_indices, np.ones(np.shape(matched_indices)))
    matched_indices = matched_indices.astype(int)    
    
    D1_matched_timecourse_network = D1_file[list(matched_indices[:,0])]
    D9_matched_timecourse_network = D9_file[list(matched_indices[:,1])]
    
    np.save(mouse_id + '_D1_matched_traces.npy' , D1_matched_timecourse_network)
    np.save(mouse_id + '_D9_matched_traces.npy' , D9_matched_timecourse_network)
    
