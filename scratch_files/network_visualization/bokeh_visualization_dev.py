"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 
File Final Edit Date:

Description: 
"""
# Import packages
from bokeh.io import show, save
from bokeh.models import Range1d, Circle, MultiLine
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.palettes import *
from bokeh.transform import linear_cmap

import networkx
from dg_graph import DGGraph as nng
import numpy as np
import os
from setup import FC_DATA_PATH

# WT Data
subject_1 = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-1_D9_smoothed_calcium_traces.csv']
subject_2 = ['1055-2_D1_smoothed_calcium_traces.csv', '1055-2_D9_smoothed_calcium_traces.csv']
subject_3 = ['1055-3_D1_smoothed_calcium_traces.csv', '1055-3_D9_smoothed_calcium_traces.csv']
subject_4 = ['1055-4_D1_smoothed_calcium_traces.csv', '1055-4_D9_smoothed_calcium_traces.csv']
subject_5 = ['14-0_D1_smoothed_calcium_traces.csv', '14-0_D9_smoothed_calcium_traces.csv']
subject_6 = ['122-1_D1_smoothed_calcium_traces.csv', '122-1_D9_smoothed_calcium_traces.csv']
subject_7 = ['122-2_D1_smoothed_calcium_traces.csv', '122-2_D9_smoothed_calcium_traces.csv']
subject_8 = ['122-3_D1_smoothed_calcium_traces.csv', '122-3_D9_smoothed_calcium_traces.csv']
subject_9 = ['122-4_D1_smoothed_calcium_traces.csv', '122-4_D9_smoothed_calcium_traces.csv']

WT_data = [subject_1, subject_2, subject_3, subject_4, subject_5, subject_6, subject_7, subject_8, subject_9]

# Th Data
subject_1 = ['348-1_D1_smoothed_calcium_traces.csv', '348-1_D9_smoothed_calcium_traces.csv']
subject_2 = ['349-2_D1_smoothed_calcium_traces.csv', '349-2_D9_smoothed_calcium_traces.csv']
subject_3 = ['386-2_D1_smoothed_calcium_traces.csv', '386-2_D9_smoothed_calcium_traces.csv']
subject_4 = ['387-4_D1_smoothed_calcium_traces.csv', '387-4_D9_smoothed_calcium_traces.csv']
subject_5 = ['396-1_D1_smoothed_calcium_traces.csv', '396-1_D9_smoothed_calcium_traces.csv']
subject_6 = ['396-3_D1_smoothed_calcium_traces.csv', '396-3_D9_smoothed_calcium_traces.csv']
subject_7 = ['2-1_D1_smoothed_calcium_traces.csv', '2-1_D9_smoothed_calcium_traces.csv']
subject_8 = ['2-2_D1_smoothed_calcium_traces.csv', '2-2_D9_smoothed_calcium_traces.csv']
subject_9 = ['2-3_D1_smoothed_calcium_traces.csv', '2-3_D9_smoothed_calcium_traces.csv']

Th_data = [subject_1, subject_2, subject_3, subject_4, subject_5, subject_6, subject_7, subject_8, subject_9]

day_idx=0
data = Th_data
data = FC_DATA_PATH  + subject_2[day_idx]
# data = WT_data
threshold = 0.3

filename = data
mouse_id = filename.strip('_smoothed_calcium_traces.csv')
nn = nng(filename)
print(f"Executing analyses for {mouse_id}")
num_neurons = nn.num_neurons

def interactive_network(show=True, save_plot=False, save_path=None):
    """
    Generates an interactived Bokeh.io plot of the graph network.


    """

for con_idx in [0, 1]:
        conA = nn.get_context_A_graph(threshold=threshold)
        conB = nn.get_context_B_graph(threshold=threshold)
        contexts = ['context_A', 'context_B']
        context = contexts[con_idx]
        context_shorthands = ['_con_A', '_con_B']
        context_shorthand = context_shorthands[con_idx]
        context_graphs = [conA, conB]
        G = context_graphs[con_idx]

            # Compute the degree of each node and add attribute
        degrees = dict(networkx.degree(G))
        networkx.set_node_attributes(G, name='degree', values=degrees)

        # # Add context active attribute
        label_keys = list(map(str, list(nn.labels)))
        con_act = list(np.genfromtxt(mouse_id + '_neuron_context_active.csv',
                                 delimiter=','))  # load context active designations
        con_act_dict: dict = {i: j for i, j in zip(label_keys, con_act)}
        networkx.set_node_attributes(G, name='context_activity', values=con_act_dict)

        # Add HITS attribute
        hub_list, hit_vals = nn.get_hubs(graph=G)
        networkx.set_node_attributes(G, name='HITS', values=hit_vals)

            # Add hubs attribute
        hub_dict: dict = {i: 1 if i in list(set(hub_list) & set(label_keys)) else 0 for i in label_keys}
        networkx.set_node_attributes(G, name='hubs', values=hub_dict)

        # Add correlated pairs attribute
        label_keys = list(map(str, list(nn.labels)))
        corr_pair = nn.get_correlated_pair_ratio(graph=G, threshold=threshold)
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
        color_by_this_attribute = 'context_activity'

        # Pick a color palette — Blues8, Reds8, Purples8, Oranges8, Viridis8
        if con_idx == 0:
            color_palette = Reds8
        else:
            color_palette = Blues8

        # use palette for context active
        color_palette = ('grey', 'red', 'blue')

        # Choose a title!
        title = f"Mouse {mouse_id} in {context} (WT)"

        # Establish which categories will appear when hovering over each node
        HOVER_TOOLTIPS = [
            ("Neuron", "@index"),
            ("Degree", "@degree"),
            #("Context Active", "@context_activity"),
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
        if show:
            show(plot)
        # Uncomment line below to save plots
        if save_plot:
            if save_path is not None:
                save(plot, filename=save_path)
            else:
                save(plot, filename=os.path.join(os.getcwd(), f"visualization/20210422/context_active_{mouse_id}{context_shorthand}.html"))
