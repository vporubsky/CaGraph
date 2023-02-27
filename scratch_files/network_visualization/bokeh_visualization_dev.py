"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 
File Final Edit Date:

Description: 
"""
# Import bokeh utilities
from bokeh.io import show, save
from bokeh.models import Range1d, Circle, MultiLine
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.palettes import *
from bokeh.transform import linear_cmap

# Import
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

def interactive_network(ca_graph_obj, graph=None, attributes = ['degree', 'HITS', 'hubs', 'CPR', 'communities'],
                        adjust_node_size = 5, adjust_size_by = 'degree', adjust_color_by = 'communities', palette = Blues8,
                        hover_attributes = ['degree', 'HITS', 'hubs', 'CPR', 'communities'], title= None,
                        show_plot=True, save_plot=False, save_path=None):
    """
    Generates an interactived Bokeh.io plot of the graph network.

    palette: a color palette which can be passed as a tuple: palette = ('grey', 'red', 'blue')

    """
    # initialize graph information
    nn = ca_graph_obj
    if graph is None:
        G = nn.get_network_graph()
    else:
        G = graph
    label_keys = list(map(str, list(nn.labels)))

    #  Build attributes dictionary
    attribute_dict = {}
    for attribute in attributes:
        if attribute == 'degree':
            # Compute the degree of each node and add attribute
            attribute_dict['degree'] = dict(networkx.degree(G))
        elif attribute == 'HITS':
            # Add HITS attribute
            hub_list, hit_vals = nn.get_hubs(graph=G)
            attribute_dict['HITS'] = hit_vals
        elif attribute == 'hubs':
            # Add hubs attribute
            attribute_dict['hubs'] = {i: 1 if i in list(set(hub_list) & set(label_keys)) else 0 for i in label_keys}
        elif attribute == 'CPR':
            # Add correlated pairs attribute
            corr_pair = nn.get_correlated_pair_ratio(graph=G, threshold=threshold)
            attribute_dict['CPR'] = {i: j for i, j in zip(label_keys, corr_pair)}
        elif attribute == 'communities':
            # Add communities
            c = list(networkx.algorithms.community.greedy_modularity_communities(G))
            sorted(c)
            community_id = {}
            for i in range(len(c)):
                for j in list(c[i]):
                    community_id[j] = i
            attribute_dict['communities'] = community_id
        else:
            raise AttributeError('Invalid attribute key entered.')

    # Set node attributes
    for key, value in attribute_dict.items():
        networkx.set_node_attributes(G, name=key, values=value)

    # Todo: make this flexible for labeled conditions
    # # Add context active attribute
    # con_act = list(np.genfromtxt(mouse_id + '_neuron_context_active.csv', delimiter=','))  # load context active designations
    # con_act_dict: dict = {i: j for i, j in zip(label_keys, con_act)}
    # networkx.set_node_attributes(G, name='context_activity', values=con_act_dict)


    # Adjusted node size
    if adjust_node_size is not None:
        # Adjust node size
        # Todo:adjusted node size not working
        adjusted_node_size = dict([(node, value + adjust_node_size) for node, value in attribute_dict[adjust_size_by]])
        networkx.set_node_attributes(G, name='adjusted_node_size', values=adjusted_node_size)
        size_by_this_attribute = 'adjusted_node_size'

    # Adjust node color
    color_by_this_attribute = adjust_color_by

    # Pick a color palette — Blues8, Reds8, Purples8, Oranges8, Viridis8
    color_palette = palette

    # Establish which categories will appear when hovering over each node
    HOVER_TOOLTIPS = [("Neuron", "@index")]
    for value in hover_attributes:
        HOVER_TOOLTIPS.append((value, "@" + value))

    # Create a plot with set dimensions, toolbar, and title
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


    # Set edge opacity and width
    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)

    plot.renderers.append(network_graph)
    if show_plot:
        show(plot)

    if save_plot:
        if save_path is not None:
            save(plot, filename=save_path)
        else:
            save(plot, filename=os.path.join(os.getcwd(), f"bokeh_graph_visualization.html"))


#%%% example
# for con_idx in [0, 1]:
#         conA = nn.get_context_A_graph(threshold=threshold)
#         conB = nn.get_context_B_graph(threshold=threshold)
#         contexts = ['context_A', 'context_B']
#         context = contexts[con_idx]
#         context_shorthands = ['_con_A', '_con_B']
#         context_shorthand = context_shorthands[con_idx]
#         context_graphs = [conA, conB]
#         G = context_graphs[con_idx]

interactive_network(ca_graph_obj=nn,adjust_size_by='CPR')