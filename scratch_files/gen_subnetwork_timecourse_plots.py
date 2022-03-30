from neuronal_network_graph import neuronal_network_graph as nng

filename = '1055-2_D1_smoothed_calcium_traces.csv'


#filename = '14-0_D1_smoothed_calcium_traces.csv'
nn = nng(filename)
threshold = 0.3

nn.plot_subnetworks_A_timecourses(threshold=threshold)
nn.plot_subnetworks_B_timecourses(threshold=threshold)