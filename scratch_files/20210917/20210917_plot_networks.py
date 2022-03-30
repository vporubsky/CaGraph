"""https://melaniewalsh.github.io/Intro-Cultural-Analytics/Network-Analysis/Making-Network-Viz-with-Bokeh.html"""
from bokeh.io import output_notebook, show, save
from bokeh.models import Range1d, Circle, MultiLine, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.palettes import Blues8, Reds8, Purples8, Oranges8, Viridis8, Spectral8
from bokeh.transform import linear_cmap
import networkx
from neuronal_network_graph import BLANetworkGraph as nng
import os
import numpy as np

ca_data = ['119-0_deconTrace.csv', '120-0_deconTrace.csv', '120-1_deconTrace.csv', \
           '519-0_deconTrace.csv', '568-4_deconTrace.csv', '568-5_deconTrace.csv', \
           '651-0_deconTrace.csv', '658-0_deconTrace.csv']
ca_data = ['120-1_deconTrace.csv']

threshold = 0.3



for subject in ca_data:
    filename = subject
    mouse_id = filename.strip('_deconTrace.csv')

    nn = nng(filename)
    print(f"Executing analyses for {mouse_id}")
    num_neurons = nn.num_neurons

    file = mouse_id + '_closed.csv'
    data = np.genfromtxt(file, delimiter=",")

    # find indices in binary array where animal is in the closed portion of EZM
    indices = [index for index, element in enumerate(data) if
               element == 1]  # if element = 1 --> in closed portion of maze
    shifted_ind = indices[1:]
    ind_new = indices[0:-1]
    # ones in the following array mean you have continuous timepoints that are in closed portion of the maze
    # note:  all other values (than 1) indicate that the mouse is in the open portion of the maze
    groups = np.array(shifted_ind) - np.array(ind_new)

    # find indices where there is a value > 1
    open_ind = [index for index, element in enumerate(groups) if element > 1]
    open_indices = []
    for i in open_ind:
        open_indices.append(indices[i])

    center_bound = []
    for i in range(len(open_indices)):
        center_bound.append(open_indices[i] + groups[open_ind[i]])

    open_indices.append(indices[-1] + 1)

    # construct closed sub sample indices
    closed = [(indices[0], open_indices[0])]
    for i in range(len(center_bound)):
        closed.append((center_bound[i], open_indices[i + 1]))

    # construct exterior sub sample indices
    open = [(0, indices[0] - 1)]
    for i in range(len(open_indices) - 1):
        open.append((open_indices[i], center_bound[i] - 1))

    subsampled_graphs = nn.get_time_subsampled_graphs(subsample_indices=closed, threshold=threshold)

    # Set position
    G = subsampled_graphs[0]
    position = networkx.spring_layout(G)
    #position = networkx.circular_layout(G)

    for graph_idx in range(len(subsampled_graphs)):
        G = subsampled_graphs[graph_idx]

        # Compute the degree of each node and add attribute
        degrees = dict(networkx.degree(G))
        networkx.set_node_attributes(G, name='degree', values=degrees)

        # Add hub attribute
        hub_list, hit_vals = nn.get_hubs(graph=subsampled_graphs[graph_idx])
        networkx.set_node_attributes(G, name='HITS', values=hit_vals)

        # Add correlated pairs attribute
        label_keys = list(map(str, list(nn.labels)))
        corr_pair = nn.get_correlated_pair_ratio(graph=subsampled_graphs[graph_idx], threshold=threshold)
        correlated_pair_ratio = {i: j for i, j in zip(label_keys, corr_pair)}
        networkx.set_node_attributes(G, name='CPR', values=correlated_pair_ratio)

        # Add communities
        c = list(networkx.algorithms.community.greedy_modularity_communities(G))
        sorted(c)
        community_id = {}
        for i in range(len(c)):
            for j in list(c[i]):
                community_id[j] = i
        networkx.set_node_attributes(G, name='community', values=community_id)

        # Adjusted node size
        number_to_adjust_by = 5
        adjusted_node_size = dict([(node, degree + number_to_adjust_by) for node, degree in networkx.degree(G)])
        networkx.set_node_attributes(G, name='adjusted_node_size', values=adjusted_node_size)

        # Choose attributes from G network to size and color by — setting manual size (e.g. 10) or color (e.g. 'skyblue') also allowed
        size_by_this_attribute = 'adjusted_node_size'
        color_by_this_attribute = 'community'

        # Pick a color palette — Blues8, Reds8, Purples8, Oranges8, Viridis8
        color_palette = Blues8

        # Choose a title!
        title = f"Mouse {mouse_id}"

        # Establish which categories will appear when hovering over each node
        HOVER_TOOLTIPS = [
            ("Neuron", "@index"),
            ("Degree", "@degree"),
            ("HITS", "@HITS"),
            ("CPR", "@CPR"),
            ("Community", "@community")
        ]

        # Create a plot — set dimensions, toolbar, and title
        plot = figure(tooltips=HOVER_TOOLTIPS,
                      tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
                      x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1), title=title)

        # Create a network graph object
        network_graph = from_networkx(G, position, scale=10, center=(0, 0))

        # Set node sizes and colors according to node degree (color as spectrum of color palette)
        minimum_value_color = min(network_graph.node_renderer.data_source.data[color_by_this_attribute])
        maximum_value_color = max(network_graph.node_renderer.data_source.data[color_by_this_attribute])
        network_graph.node_renderer.glyph = Circle(size=size_by_this_attribute,
                                                   fill_color=linear_cmap(color_by_this_attribute, color_palette,
                                                                          minimum_value_color, maximum_value_color))

        # Set edge opacity and width
        network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)


        plot.renderers.append(network_graph)

        show(plot)
        #save(plot, filename=os.path.join(os.getcwd(), f"visualization/{mouse_id}}.html"))

