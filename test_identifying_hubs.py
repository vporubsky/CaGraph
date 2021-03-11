'''Notes:
nn.get_hubs() returns the identifiers of the hubs in the network
and also a list of all the nodes and their corresponding hits values.

hubs are identified using a threshold value determined for each
'''

from neuronal_network_graph import neuronal_network_graph as nng

filename = '2-1_D1_smoothed_calcium_traces.csv'

nn = nng(filename)
hubs, hits = nn.get_hubs()
print(hubs)
