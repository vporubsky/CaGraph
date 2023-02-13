from ca_graph import DGNetworkGraph
import os
import matplotlib.pyplot as plt
import networkx as nx
import logging as _log

_log.basicConfig(filename='../DG_FC_summary.log', level=_log.INFO)

# _log.info('Context A graph generated.')
path = os.getcwd() + '/LC-DG-FC-data/'
mouse_id = '2-1_D1'
mouse_data_suffix = '_smoothed_calcium_traces.csv'

#%% generate DGGraph object
nng = DGNetworkGraph(data_file=path + mouse_id + mouse_data_suffix)

# plot an example timecourse
plt.plot(nng.time, nng.neuron_dynamics[1, :])
_log.info('Plotted single neuron timecourse using CaGraph dynamics.')
nng.plot_single_neuron_timecourse(neuron_trace_number=5, title='')
_log.info('Plotted single neuron timecourse using CaGraph plotting.')
plt.show()


#%% draw networks for context A and B
mouse_id = '2-1_D1'
mouse_data_suffix = '_smoothed_calcium_traces.csv'
nng = DGNetworkGraph(data_file=path + mouse_id + mouse_data_suffix)

plt.figure(num=1, figsize=(10,10))
plt.subplot(221)
nng.draw_network(graph=nng.get_context_A_graph(), node_color='r')
plt.title('D1 context A')

plt.subplot(222)
nng.draw_network(graph=nng.get_context_B_graph())
plt.title('D1 context B')

mouse_id = '2-1_D9'
mouse_data_suffix = '_smoothed_calcium_traces.csv'
nng = DGNetworkGraph(data_file=path + mouse_id + mouse_data_suffix)

plt.subplot(223)
nng.draw_network(graph=nng.get_context_A_graph(), node_color='r')
plt.title('D9 context A')
plt.subplot(224)
nng.draw_network(graph=nng.get_context_B_graph())
plt.title('D9 context B')
plt.show()
_log.info('Drew context A and B networks on day 1 and 9.')

#%% Plot weighted network for plot_metrics
plt.figure()
plt.subplot(121)
G = nng.get_network_graph_from_matrix(weight_matrix=nng.con_A_pearsons_correlation_matrix)
elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] >= 0.3]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] < 0.3]

pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

# nodes
nx.draw_networkx_nodes(G, pos, node_size=30)

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge)
nx.draw_networkx_edges(
    G, pos, edgelist=esmall, alpha=0.1, edge_color="b", style="dashed"
)
plt.title('Context A')


plt.subplot(122)
G = nng.get_network_graph_from_matrix(weight_matrix=nng.con_B_pearsons_correlation_matrix)
elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] >= 0.3]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] < 0.3]


# nodes
nx.draw_networkx_nodes(G, pos, node_size=30, node_color='r')

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge)
nx.draw_networkx_edges(
    G, pos, edgelist=esmall, alpha=0.1, edge_color="r", style="dashed"
)
plt.title('Context B')
plt.show()

#%% show weighted network hits histogram
mouse_id = '2-2_D1'
mouse_data_suffix = '_smoothed_calcium_traces.csv'
nng = DGNetworkGraph(data_file=path + mouse_id + mouse_data_suffix)

# Day 1
plt.figure(figsize=(10,10))
plt.subplot(221)
weighted_A = nng.get_network_graph_from_matrix(weight_matrix=nng.con_A_pearsons_correlation_matrix)
hits = nx.hits_numpy(weighted_A)
hits_val = list(hits[0].values())
plt.xlim(-0.05, 0.05)
plt.hist(hits_val, color='r')
plt.title('Day 1 Context A')

plt.subplot(222)
weighted_B = nng.get_network_graph_from_matrix(weight_matrix=nng.con_B_pearsons_correlation_matrix)
hits = nx.hits_numpy(weighted_B)
hits_val = list(hits[0].values())
plt.hist(hits_val, color='b')
plt.xlim(-0.05, 0.05)
plt.title('Day 1 Context B')

# Day 9
mouse_id = '2-2_D9'
mouse_data_suffix = '_smoothed_calcium_traces.csv'
nng = DGNetworkGraph(data_file=path + mouse_id + mouse_data_suffix)

plt.subplot(223)
weighted_A = nng.get_network_graph_from_matrix(weight_matrix=nng.con_A_pearsons_correlation_matrix)
hits = nx.hits_numpy(weighted_A)
hits_val = list(hits[0].values())
plt.xlim(-0.05, 0.05)
plt.hist(hits_val, color='r')
plt.title('Day 9 Context A')

plt.subplot(224)
weighted_B = nng.get_network_graph_from_matrix(weight_matrix=nng.con_B_pearsons_correlation_matrix)
hits = nx.hits_numpy(weighted_B)
hits_val = list(hits[0].values())
plt.hist(hits_val, color='b')
plt.xlim(-0.05, 0.05)
plt.title('Day 9 Context B')

plt.suptitle(mouse_id.strip('_D9'))
plt.show()

#%% communities with plotting by node colored according to community
G = nng.get_context_A_graph()
communities = nng.get_communities(graph=G)
communities.sort(key=len)
communities.reverse()

color_list = ['xkcd:sky blue', 'xkcd:cadet blue', 'xkcd:sea blue', 'xkcd:bright blue', 'blue']
color_map = []
num_communities = 3


long_list = []
for community_idx in range(num_communities - 1):
    long_list = long_list + communities[community_idx]

for node in G:
    for community_idx in range(num_communities - 1):
        if node in communities[community_idx]:
            color_map.append(color_list[community_idx])
    if node not in long_list:
        color_map.append(color_list[num_communities - 1])

nx.draw(G, node_size = 50, node_color=color_map, alpha=0.5)
plt.show()
