# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 14:29:14 2020

@author: Veronica Porubsky

Title: test graph theory breakdown
"""
import logging
from neuronal_network_graph import neuronal_network_graph as nng

# Configure hyperparameters
threshold = 0.5

# set up logger
logging.basicConfig(filename='log_graph_theory_limit_05.log',format='%(asctime)s %(name)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S', filemode='w')
_log = logging.getLogger('__graph_theory_limit_log__')
_log.info('Begin logging...')

#%% Load untreated data files
day1_untreated = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv', '1055-3_D1_smoothed_calcium_traces.csv', '1055-4_D1_smoothed_calcium_traces.csv', '14-0_D1_smoothed_calcium_traces.csv']
day5_untreated = ['1055-1_D5_smoothed_calcium_traces.csv', '1055-2_D5_smoothed_calcium_traces.csv','1055-3_D5_smoothed_calcium_traces.csv', '1055-4_D5_smoothed_calcium_traces.csv', '14-0_D5_smoothed_calcium_traces.csv']
day9_untreated = ['1055-1_D9_smoothed_calcium_traces.csv', '1055-2_D9_smoothed_calcium_traces.csv','1055-3_D9_smoothed_calcium_traces.csv', '1055-4_D9_smoothed_calcium_traces.csv', '14-0_D9_smoothed_calcium_traces.csv']

all_files = [day1_untreated, day5_untreated, day9_untreated]
_log.info(f'Analyses performed with edge weight threshold: {threshold}')

for treatment_group_index in [0, 1, 2]:

    # Log recording day
    if treatment_group_index == 0:
        _log.info(f'--- Analyzing Day 1 ---')
    elif treatment_group_index == 1:
        _log.info(f'--- Analyzing Day 5 ---')
    else:
        _log.info(f'--- Analyzing Day 9 ---')

    # Begin analyses
    for mouse_id_index in range(len(all_files[treatment_group_index])):

        analysis_successful = True
        filename = all_files[treatment_group_index][mouse_id_index]
        mouse_id = filename.strip('_smoothed_calcium_traces.csv')

        nn = nng(filename)
        print(f"Executing analyses for {mouse_id}")
        num_neurons = nn.num_neurons

        specimen_dict = {"mouse_id": mouse_id, "succeeded": [], "failed": []}

        conA = nn.get_context_A_graph(threshold=threshold)
        conB = nn.get_context_B_graph(threshold=threshold)

        # subnetworks
        connected_subnetworks_A = nn.get_context_A_subnetworks(threshold=threshold)
        connected_subnetworks_B = nn.get_context_B_subnetworks(threshold=threshold)

        num_connected_subnetworks_A = len(connected_subnetworks_A)
        len_connected_subnetworks_A = []
        [len_connected_subnetworks_A.append(len(x)) for x in connected_subnetworks_A]

        num_connected_subnetworks_B = len(connected_subnetworks_B)
        len_connected_subnetworks_B = []
        [len_connected_subnetworks_B.append(len(x)) for x in connected_subnetworks_B]

        if num_connected_subnetworks_A == 0:
            analysis_successful = False
            specimen_dict["failed"].append('subnetwork count: A')
        else:
            specimen_dict["succeeded"].append('subnetwork count: A')

        if num_connected_subnetworks_B == 0:
            analysis_successful = False
            specimen_dict["failed"].append('subnetwork count: B')
        else:
            specimen_dict["succeeded"].append('subnetwork count: B')

        hubs_A = nn.get_context_A_hubs(threshold=threshold)
        hubs_B = nn.get_context_B_hubs(threshold=threshold)

        len_hubs_A = len(hubs_A)
        len_hubs_B = len(hubs_B)

        if len_hubs_A == 0:
            analysis_successful = False
            specimen_dict["failed"].append('hub count: A')
        else:
            specimen_dict["succeeded"].append('hub count: A')

        if len_hubs_B == 0:
            analysis_successful = False
            specimen_dict["failed"].append('hub count: B')
        else:
            specimen_dict["succeeded"].append('hub count: B')

        if not analysis_successful:
            _log.info(f'{mouse_id} (neurons: {num_neurons}, con_A edges: {len(conA.edges())}, con_B edges: {len(conB.edges())}) analyses failed: {str(specimen_dict["failed"])}.')
        else:
            _log.info(f'{mouse_id} (neurons: {num_neurons}, con_A edges: {len(conA.edges())}, con_B edges: {len(conB.edges())}) all analyses successful.')


_log.info('End logging.')