'''Verifying results of test_graph_theory_limit.py based on logs'''
from neuronal_network_graph import neuronal_network_graph as nng

# Configure hyperparameters
threshold = 0.7

#%% Load untreated data files
day1_untreated = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv', '1055-3_D1_smoothed_calcium_traces.csv', '1055-4_D1_smoothed_calcium_traces.csv', '14-0_D1_smoothed_calcium_traces.csv']
day9_untreated = ['1055-1_D9_smoothed_calcium_traces.csv', '1055-2_D9_smoothed_calcium_traces.csv','1055-3_D9_smoothed_calcium_traces.csv', '1055-4_D9_smoothed_calcium_traces.csv', '14-0_D9_smoothed_calcium_traces.csv']

nn = nng(day1_untreated[4])

conA = nn.get_context_A_graph(threshold=threshold)
conB = nn.get_context_B_graph(threshold=threshold)

connected_subnetworks_A = nn.get_context_A_subnetworks(threshold=threshold)
connected_subnetworks_B = nn.get_context_B_subnetworks(threshold=threshold)