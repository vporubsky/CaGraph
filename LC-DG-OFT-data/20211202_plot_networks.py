"""https://melaniewalsh.github.io/Intro-Cultural-Analytics/Network-Analysis/Making-Network-Viz-with-Bokeh.html"""
from bokeh.io import output_notebook, show, save
from bokeh.models import Range1d, Circle, MultiLine, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.palettes import Blues8, Reds8
from bokeh.transform import linear_cmap
import networkx
from ca_graph import DGNetworkGraph as nng
import os
import numpy as np

data_198_1 = ['198-1_Saline.csv', '198-1_Prop.csv', '198-1_Praz.csv', '198-1_Quetiapine5mg-kg.csv', '198-1_Quetiapine10mg-kg.csv',
            '198-1_CNOSaline.csv', '198-1_CNOPrazosin.csv', '198-1_CNOQuetiapine5mg-kg.csv']

data_202_4 = ['202-4_Saline.csv', '202-4_Prop.csv', '202-4_Praz.csv', '202-4_Quetiapine5mg-kg.csv', '202-4_Quetiapine10mg-kg.csv',
            '202-4_CNOSaline.csv', '202-4_CNOPrazosin.csv', '202-4_CNOQuetiapine5mg-kg.csv']

ca_data = data_202_4

threshold = 0.4


for subject in ca_data:
    filename = subject
    mouse_id = filename[0:5]
    nn = nng(filename)
    print(f"Executing analyses for {mouse_id}")
    num_neurons = nn.num_neurons

    # To set single position keep lines below uncommented
    # G = nn.get_context_A_graph()
    # position = networkx.spring_layout(G)

    for con_idx in [0, 1]:
        subsampled_graphs = nn.get_time_subsampled_graphs(subsample_indices=[(0, 360), (360, 3600)],
                                                          threshold=threshold)
        conA = subsampled_graphs[0]
        conB = subsampled_graphs[1]
        contexts = ['context_A', 'context_B']
        context = contexts[con_idx]
        context_shorthands = ['_con_A', '_con_B']
        context_shorthand = context_shorthands[con_idx]
        context_graphs = [conA, conB]
        G = context_graphs[con_idx]

        # Compute the degree of each node and add attribute
        degrees = dict(networkx.degree(G))
        networkx.set_node_attributes(G, name='degree', values=degrees)

        # Add context active attribute
        # label_keys = list(map(str, list(nn.labels)))
        # con_act = list(np.genfromtxt(os.getcwd() + '/' + mouse_id + '_neuron_context_active.csv',
        #                              delimiter=','))  # load context active designations
        # con_act_dict: dict = {i: j for i, j in zip(label_keys, con_act)}
        # networkx.set_node_attributes(G, name='context_activity', values=con_act_dict)

        # Add HITS attribute
        hub_list, hit_vals = nn.get_hubs(graph=G)
        networkx.set_node_attributes(G, name='HITS', values=hit_vals)

        # # Add hubs attribute
        # hub_dict: dict = {i: 1 if i in list(set(hub_list) & set(label_keys)) else 0 for i in label_keys}
        # networkx.set_node_attributes(G, name='hubs', values=hub_dict)
        #
        # # Add correlated pairs attribute
        # label_keys = list(map(str, list(nn.labels)))
        # corr_pair = nn.get_correlated_pair_ratio(graph=G, threshold=threshold)
        # correlated_pair_ratio = {i: j for i, j in zip(label_keys, corr_pair)}
        # networkx.set_node_attributes(G, name='CPR', values=correlated_pair_ratio)

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
        if con_idx == 0:
            color_palette = Reds8
        else:
            color_palette = Blues8

        # use palette for context active
        color_palette = Blues8

        # Choose a title!
        title = f"Mouse {mouse_id} in {context} (WT)"

        # Establish which categories will appear when hovering over each node
        HOVER_TOOLTIPS = [
            ("Neuron", "@index"),
            ("Degree", "@degree"),
            ("Context Active", "@context_activity"),
            ("HITS", "@HITS"),
            ('Hub', "@hubs"),
            ("CPR", "@CPR"),
            ("Community", "@community")
        ]

        # Create a plot — set dimensions, toolbar, and title
        plot = figure(tooltips=HOVER_TOOLTIPS,
                      tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
                      x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1), title=title)

        # Create a network graph object
        position = networkx.spring_layout(G)
        network_graph = from_networkx(G, position, scale=10, center=(0, 0))

        # Set node sizes and colors according to node degree (color as spectrum of color palette)
        minimum_value_color = min(network_graph.node_renderer.data_source.data[color_by_this_attribute])
        maximum_value_color = max(network_graph.node_renderer.data_source.data[color_by_this_attribute])
        network_graph.node_renderer.glyph = Circle(size=size_by_this_attribute,
                                                   fill_color=linear_cmap(color_by_this_attribute, color_palette,
                                                                          minimum_value_color, maximum_value_color))
        # network_graph.node_renderer.selection_glyph = Circle(size=15, fill_color=Spectral4[2])
        # network_graph.node_renderer.hover_glyph = Circle(size=15, fill_color=Spectral4[1])

        # Set edge opacity and width
        network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)
        # network_graph.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[2], line_width=5)
        # network_graph.edge_renderer.hover_glyph = MultiLine(line_color=Spectral4[1], line_width=5)
        #
        # network_graph.selection_policy = NodesAndLinkedEdges()
        # network_graph.inspection_policy = EdgesAndLinkedNodes()

        plot.renderers.append(network_graph)
        show(plot)
        # Uncomment line below to save plots
        #save(plot, filename=os.path.join(os.getcwd(), f"plot_metrics/20210422/context_active_{mouse_id}{context_shorthand}.html"))
