'''Test functionality update on 12-18-2020'''
from neuronal_network_graph import neuronal_network_graph as nng
import numpy as np
filename = '1055-2_D9_smoothed_calcium_traces.csv'

nn = nng(filename)
cpr = nn.get_correlated_pair_ratio()
cc = nn.get_clustering_coefficient()
cc_A = nn.get_context_A_clustering_coefficient()
cc_B = nn.get_context_B_clustering_coefficient()

print(f'clustering coeff for context A is: {np.average(cc_A)}')
print(f'clustering coeff for context B is: {np.average(cc_B)}')