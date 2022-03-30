'''Smallworld analyses failing, testing the networkx implementation directly.'''

from neuronal_network_graph import neuronal_network_graph as nng
import networkx as nx
import numpy as np

filename = '1055-2_D1_smoothed_calcium_traces.csv'
nn = nng(filename)
threshold = 0.5

mouse_id = filename.strip('_smoothed_calcium_traces.csv')

conA = nn.get_context_A_graph(threshold=threshold)
conB = nn.get_context_B_graph(threshold=threshold)

# small-world analysis
con_A_omega_val = nn.get_smallworld_largest_subnetwork(graph=conA, threshold=threshold)
print(f'Completed analysis of {mouse_id} context A with omega val: {con_A_omega_val}')
con_B_omega_val = nn.get_smallworld_largest_subnetwork(graph=conB, threshold=threshold)
print(f'Completed analysis of {mouse_id} context B with omega val: {con_B_omega_val}')


#%% Troubleshooting using code directly from networkx package
# nrand = 10
# niter = 100
# seed = None
# randMetrics = {"C": [], "L": []}
# for i in range(nrand):
#     Gr = nx.smallworld.random_reference(nn.get_largest_context_A_subnetwork_graph(threshold=threshold), niter=niter, seed=seed)
#     Gl = nx.smallworld.lattice_reference(nn.get_largest_context_A_subnetwork_graph(threshold=threshold), niter=niter, seed=seed)
#     randMetrics["C"].append(nx.transitivity(Gl))
#     randMetrics["L"].append(nx.average_shortest_path_length(Gr))
#     print(f'iteration: {i}')
#
#
# C = nx.transitivity(nn.get_largest_context_A_subnetwork_graph(threshold=threshold))
# L = nx.average_shortest_path_length(nn.get_largest_context_A_subnetwork_graph(threshold=threshold))
# Cl = np.mean(randMetrics["C"])
# Lr = np.mean(randMetrics["L"])
#
# omega = (Lr / L) - (C / Cl)


# # Todo: test generating subcomponent graphs
# connected_components = list(nx.connected_components(nn.get_network_graph(threshold = threshold)))
# G = nn.get_network_graph(threshold = threshold)
# S = [G.subgraph(c).copy() for c in nx.connected_components(G) if len(c) > 1] # subgraph must contain more than a single node
#
# # Todo: return only the largest subcomponent graph:
# G = nn.get_context_B_graph(threshold = threshold)
# largest_component = max(nx.connected_components(G), key = len)
# subgraph_lc = G.subgraph(largest_component)
# print(len(subgraph_lc.nodes()))