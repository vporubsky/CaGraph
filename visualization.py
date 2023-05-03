"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 

Description: 
"""
# General imports
from cagraph import CaGraph
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('CaGraph.mplstyle')
import seaborn as sns
from scipy import stats
import pandas as pd
import os

# Bokeh imports
import bokeh
from bokeh.io import show, save, output_notebook
from bokeh.models import Range1d, Circle, MultiLine
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.transform import linear_cmap

def interactive_network(cagraph_obj, graph=None,
                        attributes=['degree', 'HITS', 'hubs', 'CPR', 'communities', 'clustering'],
                        additional_attributes = None,
                        hover_attributes=None,
                        adjust_node_size=5, adjust_size_by='degree', adjust_color_by='communities',
                        palette='Blues8',
                        position = None, return_position = False,
                        title=None, show_plot=True, show_in_notebook=False, save_plot=False, save_path=None):
    """
    Generates an interactive Bokeh.io plot of the graph network.

    :param: cagraph_obj: CaGraph object
    :param: graph: networkx.Graph object
    :param: attributes: list
    :param: additional_attributes: dict
    :param: hover_attributes: list
    :param: adjust_node_size: int
    :param: adjust_size_by: str
    :param: adjust_color_by: str
    :param: palette: tuple a color palette which can be passed as a tuple: palette = ('grey', 'red', 'blue')
    :param: position: dict
    :param: return_position: bool
    :param: title: str
    :param: show_plot: bool
    :param: show_in_notebook: bool
    :param: save_plot: bool
    :param: save_path: str

    """
    # Initialize cagraph and graph
    cg = cagraph_obj
    if graph is None:
        G = cg.graph
    else:
        G = graph
    label_keys = cg.labels

    #  Build attributes dictionary
    attribute_dict = {}
    for attribute in attributes:

        if attribute == 'degree':
            # Compute the degree of each node and add attribute
            attribute_dict['degree'] = cg.graph_theory.get_degree()

        elif attribute == 'HITS':
            # Add HITS attribute
            attribute_dict['HITS'] = cg.graph_theory.get_hits_values()

        elif attribute == 'hubs':
            # Add hubs attribute
            attribute_dict['hubs'] = cg.graph_theory.get_hubs()

        elif attribute == 'CPR':
            # Add correlated pairs attribute
            attribute_dict['CPR'] = cg.graph_theory.get_correlated_pair_ratio()

        elif attribute == 'communities':
            # Add communities
            c = list(nx.algorithms.community.greedy_modularity_communities(G))
            sorted(c)
            community_id = {}
            for i in range(len(c)):
                for j in list(c[i]):
                    community_id[j] = i
            attribute_dict['communities'] = community_id

        elif attribute == 'clustering':
            # Add clustering coefficient
            attribute_dict['clustering'] = cg.graph_theory.get_clustering_coefficient()

        else:
            raise AttributeError('Invalid attribute key entered.')

    add_hover_attributes = []
    if additional_attributes is not None:
        for key in additional_attributes.keys():
            # parse attribute
            attribute_dict[key] = {i: j for i, j in zip(label_keys, additional_attributes[key])} # Must be same length
            # store key value
            add_hover_attributes += [key]

    # Set node attributes
    for key, value in attribute_dict.items():
        nx.set_node_attributes(G, name=key, values=value)

    # Adjusted node size
    if adjust_node_size is not None:
        # Adjust node size
        adjusted_node_size = dict(
            [(node, value + adjust_node_size) for node, value in attribute_dict[adjust_size_by].items()])
        nx.set_node_attributes(G, name='adjusted_node_size', values=adjusted_node_size)
        size_by_this_attribute = 'adjusted_node_size'

    # Adjust node color
    color_by_this_attribute = adjust_color_by

    # Generate color palette
    palettes = [attr for attr in dir(bokeh.palettes) if
                not callable(getattr(bokeh.palettes, attr)) and not attr.startswith("__")]
    if isinstance(palette, str) and palette in palettes:
        color_palette = getattr(bokeh.palettes, palette)
    elif isinstance(palette, tuple):
        color_palette = palette
    else:
        raise AttributeError('Must specify color palette as type string using an existing bokeh.palettes palette or generate a tuple containing hex codes.')

    # Establish which categories will appear when hovering over each node
    if hover_attributes is None:
        hover_attributes = ['degree', 'HITS', 'hubs', 'CPR', 'communities', 'clustering'] + add_hover_attributes
    HOVER_TOOLTIPS = [("Neuron", "@index")]
    for value in hover_attributes:
        HOVER_TOOLTIPS.append((value, "@" + value))

    # Create a plot with set dimensions, toolbar, and title
    plot = figure(tooltips=HOVER_TOOLTIPS,
                  tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
                  x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1), title=title)

    # Create a network graph object
    if position is None:
        position = nx.spring_layout(G)
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
    if show_in_notebook:
        output_notebook()
        show(plot)
    elif show_plot:
        show(plot)

    if save_plot:
        if save_path is not None:
            save(plot, filename=save_path)
        else:
            save(plot, filename=os.path.join(os.getcwd(), f"bokeh_graph_visualization.html"))

    if return_position:
        return position

# Todo: make flexible for multiple datasets and remove plot_CDFs() function
def plot_CDF(data=None, color='black', marker='o', x_label='', y_label='CDF', show_plot=True, save_plot=False, save_path=None, dpi=300, format='png'):
    """
    Plots the cumulative distribution function of the provided list of data.

    :param data: list of float values
    :param color: str matplotlib color style
    :param marker: str matplotlib marker style
    :param x_label: str
    :param y_label: str
    :param show_plot: bool
    """

    # sort the dataset in ascending order
    sorted_data = np.sort(data)

    # get the cdf values of dataset
    cdf = np.arange(len(data)) / float(len(data))

    # plotting
    plt.plot(sorted_data, cdf, color=color, marker=marker)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if show_plot:
        plt.show()

    if save_plot:
        if save_path is None:
            save_path = os.getcwd() + f'fig'
        plt.savefig(fname=save_path, dpi=dpi, format=format)


def plot_CDFs(data_list=None, colors=['black', 'black'], marker='o', x_label='',
                                 y_label='CDF', legend=None, show_plot=True, save_plot=False, save_path=None, dpi=300, format='png'):
    """
    Plots the cumulative distribution function of the provided datasets and prints the associated P-value for assessing
    the Kolmogorov-Smirnov distance between the distributions.

    :param data_list: list of lists containing float values to compare with KS-test
    :param colors: list of str containing matplotlib color styles
    :param marker: str matplotlib marker style
    :param x_label: str
    :param y_label: str
    :param show_plot: bool
    """
    # Evaluate KS-test statistic
    stat_level = stats.ks_2samp(data_list[0], data_list[1])

    for idx, data in enumerate(data_list):
        # sort the dataset in ascending order
        sorted_data = np.sort(data)

        # get the cdf values of dataset
        cdf = np.arange(len(data)) / float(len(data))

        # plotting
        plt.plot(sorted_data, cdf, color=colors[idx], marker=marker)

    if legend is not None:
        plt.legend([legend[0], legend[1]])
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(f'P value: {stat_level.pvalue:.2e}')

    if show_plot:
        plt.show()

    if save_plot:
        if save_path is None:
            save_path = os.getcwd() + f'fig'
        plt.savefig(fname=save_path, dpi=dpi, format=format)

# Todo: harmonize with preprocess version
def plot_histograms(data_list=None, colors=['black', 'black'], x_label='',
                    y_label='count', bin_size=20, show_plot=True, save_plot=False, save_path=None, dpi=300, format='png'):
    """
    Plots histograms of the provided data and prints the associated P-value for assessing
    the Kolmogorov-Smirnov distance between the distributions.

    :param data_list: list of lists containing float values to compare with KS-test
    :param colors: list of str containing matplotlib color styles
    :param marker: str matplotlib marker style
    :param x_label: str
    :param y_label: str
    :param bin_size: int
    :param show_plot: bool
    :param save_plot: bool
    :param save_path: str
    :param dpi: int
    :param format: str
    """

    # Evaluate KS-test statistic
    stat_level = stats.ks_2samp(data_list[0], data_list[1])

    for idx, data in enumerate(data_list):
        # plotting
        plt.hist(data, color=colors[idx], bins=bin_size, alpha=0.4)

    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(f'P value: {stat_level.pvalue:.2e}')

    if show_plot:
        plt.show()
    if save_plot:
        if save_path is None:
            save_path = os.getcwd() + f'fig'
        plt.savefig(fname=save_path, dpi=dpi, format=format)

def plot_matched_data(sample_1: list, sample_2: list, labels: list, colors=['grey', 'grey'],
                      x_label=None, y_label=None, show_plot=True, save_plot=False, save_path=None,
                      dpi=300, format='png'):
    """
    Plots two samples of matched data. Each sample will be plotted as points stacked vertically within condition.
    Lines will be drawn to connect the matching pairs.

    :param sample_1:
    :param sample_2:
    :param colors: list of str containing matplotlib color styles
    :param x_label: str
    :param y_label: str
    :param show_plot: bool
    :param save_plot: bool
    :param dpi: int
    :param format: str containing file extension supported by matplotlib.pyplot.savefig
    """
    # Put into dataframe
    df = pd.DataFrame({labels[0]: sample_1, labels[1]: sample_2})
    data = pd.melt(df)

    # Plot
    fig, ax = plt.subplots()
    sns.swarmplot(data=data, x='variable', y='value', ax=ax, size=0)
    idx0 = 0
    idx1 = 1
    locs1 = ax.get_children()[idx0].get_offsets()
    locs2 = ax.get_children()[idx1].get_offsets()

    y_all = np.zeros(2)
    for i in range(locs1.shape[0]):
        x = [locs1[i, 0], locs2[i, 0]]
        y = [locs1[i, 1], locs2[i, 1]]
        ax.plot(x, y, color='lightgrey', linewidth=0.5)
        ax.plot(locs1[i, 0], locs1[i, 1], '.', color=colors[0])
        ax.plot(locs2[i, 0], locs2[i, 1], '.', color=colors[1])
        data = [locs1[:, 1], locs2[:, 1]]
        ax.boxplot(data, positions=[0, 1], capprops=dict(linewidth=0.5, color='k'),
                   whiskerprops=dict(linewidth=0.5, color='k'),
                   boxprops=dict(linewidth=0.5, color='k'),
                   medianprops=dict(color='k'))
        plt.xticks([])
        y_all = np.vstack((y_all, y))

    if y_label is not None:
        plt.ylabel(y_label)
    if x_label is None:
        plt.xlabel(f'P-value = {stats.ttest_rel(sample_1, sample_2).pvalue:.3}')

    if show_plot:
        plt.show()
    if save_plot:
        if save_path is None:
            save_path = os.getcwd() + f'fig'
        plt.savefig(fname=save_path, dpi=dpi, format=format)