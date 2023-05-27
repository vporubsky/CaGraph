"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

Description: a visualization module to generate interactive graph visuals and plot graph theory analyses generated using
the cagraph module.
"""
# General imports
import cagraph
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

sns.set_style('white')


# %% Interactive graph visualization

def _interactive_network_input_validator(input_object):
    """
    :param input_object:
    :return:
    """
    if isinstance(input_object, cagraph.CaGraph):
        return input_object
    else:
        raise TypeError('cagraph_obj must be type cagraph.CaGraph.')


# Todo: create return with more information
# Todo: add more base attributes
def interactive_network(cagraph_obj,
                        attributes=['degree', 'betweenness centrality', 'hubs', 'CPR', 'communities', 'clustering'],
                        additional_attributes=None,
                        hover_attributes=None,
                        adjust_node_size=5, adjust_node_size_by='degree', adjust_node_color_by='communities',
                        palette='Blues8',
                        position=None, return_position=False,
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
    :param: adjust_node_color_by: str
    :param: palette: tuple a color palette which can be passed as a tuple: palette = ('grey', 'red', 'blue')
    :param: position: dict
    :param: return_position: bool
    :param: title: str
    :param: show_plot: bool
    :param: show_in_notebook: bool
    :param: save_plot: bool
    :param: save_path: str

    """
    # Set default analyses to incorporate in interactive visualization
    default_analyses = ['degree', 'betweenness centrality', 'hubs', 'CPR', 'communities', 'clustering']

    # Initialize cagraph and graph
    cg = _interactive_network_input_validator(cagraph_obj)
    graph = cg.graph
    label_keys = cg.node_labels

    #  Build attributes dictionary
    attribute_dict = {}
    for attribute in attributes:

        if attribute == 'degree':
            # Compute the degree of each node and add attribute
            attribute_dict['degree'] = cg.graph_theory.get_degree(return_type='dict')

        elif attribute == 'betweenness centrality':
            # Add HITS attribute
            attribute_dict['betweenness centrality'] = cg.graph_theory.get_betweenness_centrality(return_type='dict')

        elif attribute == 'hubs':
            # Add hubs attribute
            attribute_dict['hubs'] = cg.graph_theory.get_hubs(return_type='dict')

        elif attribute == 'CPR':
            # Add correlated pairs attribute
            attribute_dict['CPR'] = cg.graph_theory.get_correlated_pair_ratio(return_type='dict')

        elif attribute == 'communities':
            # Add communities
            attribute_dict['communities'] = cg.graph_theory.get_communities(return_type='dict')

        elif attribute == 'clustering':
            # Add clustering coefficient
            attribute_dict['clustering'] = cg.graph_theory.get_clustering_coefficient(return_type='dict')

        else:
            raise AttributeError('Invalid attribute key entered.')

    add_hover_attributes = []
    if additional_attributes is not None:
        for key in additional_attributes.keys():
            # parse attribute
            attribute_dict[key] = {i: j for i, j in zip(label_keys, additional_attributes[key])}  # Must be same length
            # store key value
            add_hover_attributes += [key]

    # Set node attributes
    for key, value in attribute_dict.items():
        nx.set_node_attributes(graph, name=key, values=value)

    # Adjusted node size
    if adjust_node_size is not None:
        # Adjust node size
        adjusted_node_size = dict(
            [(node, value + adjust_node_size) for node, value in attribute_dict[adjust_node_size_by].items()])
        nx.set_node_attributes(graph, name='adjusted_node_size', values=adjusted_node_size)
        size_by_this_attribute = 'adjusted_node_size'

    # Adjust node color
    color_by_this_attribute = adjust_node_color_by

    # Generate color palette
    palettes = [attr for attr in dir(bokeh.palettes) if
                not callable(getattr(bokeh.palettes, attr)) and not attr.startswith("__")]
    if isinstance(palette, str) and palette in palettes:
        color_palette = getattr(bokeh.palettes, palette)
    elif isinstance(palette, tuple):
        color_palette = palette
    else:
        raise AttributeError(
            'Must specify color palette as type string using an existing bokeh.palettes palette or generate a tuple containing hex codes.')

    # Establish which categories will appear when hovering over each node
    if hover_attributes is None:
        hover_attributes = default_analyses + add_hover_attributes
    hover_tooltips = [("Neuron", "@index")]
    for value in hover_attributes:
        hover_tooltips.append((value, "@" + value))

    # Create a plot with set dimensions, toolbar, and title
    plot = figure(tooltips=hover_tooltips,
                  tools="pan,wheel_zoom,save,reset", active_scroll='wheel_zoom',
                  x_range=Range1d(-10.1, 10.1), y_range=Range1d(-10.1, 10.1), title=title)

    # Create a network graph object
    if position is None:
        position = nx.spring_layout(graph)
    network_graph = from_networkx(graph, position, scale=10, center=(0, 0))

    # Set node sizes and colors according to node degree (color as spectrum of color palette)
    minimum_value_color = min(network_graph.node_renderer.data_source.data[color_by_this_attribute])
    maximum_value_color = max(network_graph.node_renderer.data_source.data[color_by_this_attribute])
    if adjust_node_size_by is None:
        network_graph.node_renderer.glyph = Circle(fill_color=linear_cmap(color_by_this_attribute, color_palette,
                                                                          minimum_value_color, maximum_value_color))
    else:
        network_graph.node_renderer.glyph = Circle(size=size_by_this_attribute,
                                                   fill_color=linear_cmap(color_by_this_attribute, color_palette,
                                                                          minimum_value_color, maximum_value_color))

    # Set edge opacity and width
    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)

    plot.renderers.append(network_graph)
    if save_plot:
        if save_path is not None:
            save(plot, filename=save_path)
        else:
            save(plot, filename=os.path.join(os.getcwd(), f"bokeh_graph_visualization.html"))
    if show_in_notebook:
        output_notebook()
        show(plot)
    elif show_plot:
        show(plot)
    if return_position:
        return position


# %% Plotting utilities
def _plotting_input_validator(input_data):
    """

    :param input_data:
    :return:
    """
    if isinstance(input_data, list):
        for item in input_data:
            if isinstance(item, list):
                continue
            else:
                TypeError('input_data must be a list of lists containing  individual datasets for plotting.')
        return input_data
    else:
        TypeError('input_data must be a list of lists containing  individual datasets for plotting.')


def plot_cdf(data_list, colors=['black', 'black'], marker='.', xlabel='',
             ylabel='CDF', xlim=None, ylim=None, label=None, title=None, show_stat=False,
             show_plot=True, save_plot=False, save_path=None, dpi=300, save_format='png', **kwargs):
    """
    Plots the cumulative distribution function of the provided datasets and prints the associated P-value for assessing
    the Kolmogorov-Smirnov distance between the distributions.


    :param save_format:
    :param dpi:
    :param save_path:
    :param save_plot:
    :param label:
    :param ylim:
    :param xlim:
    :param data_list: list of lists containing float values to compare with KS-test
    :param colors: list of str containing matplotlib color styles
    :param marker: str matplotlib marker style
    :param xlabel: str
    :param ylabel: str
    :param title:
    :param show_stat:
    :param show_plot: bool
    """
    data_list = _plotting_input_validator(data_list)

    store_cdf = []
    for idx, data in enumerate(data_list):
        # sort the dataset in ascending order
        sorted_data = np.sort(data)

        # get the cdf values of dataset
        cdf = np.cumsum(np.histogram(data, bins=len(data))[0]) / len(data)
        store_cdf.append(cdf)

        # plotting
        plt.plot(sorted_data, cdf, color=colors[idx], marker=marker, **kwargs)

    if label is not None:
        plt.legend([label[0], label[1]], loc='upper left')

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if title is not None:
        plt.title(title)
    if show_stat:
        if len(store_cdf) != 2:
            IndexError('show_stat argument only compatible with a data_list containing two datasets.')

        # Evaluate KS-test statistic
        stat_level = stats.ks_2samp(store_cdf[0], store_cdf[1])

        # Add the text annotation below the figure legend
        if stat_level.pvalue < 0.05:
            plt.text(0.01, 0.8, f'* p-val: {stat_level.pvalue:.2e}')
        else:
            plt.text(0.01, 0.8, f'n.s.')

    sns.despine(offset=10, trim=True)
    if save_plot:
        if save_path is None:
            save_path = os.getcwd() + f'fig'
        plt.savefig(fname=save_path, bbox_inches='tight', dpi=dpi, format=save_format)
    if show_plot:
        plt.show()


def plot_histogram(data_list, colors, label=None, num_bins=None, title=None, ylabel=None, xlabel=None, alpha=0.3,
                   show_plot=True,
                   save_plot=False,
                   save_path=None, dpi=300, save_format='png', **kwargs):
    """
    Plot histograms of the provided datasets in data.

    :param alpha:
    :param num_bins:
    :param data_list: list
    :param colors: list
    :param label: list
    :param title:
    :param ylabel:
    :param xlabel:
    :param show_plot:
    :param save_plot:
    :param save_path:
    :param dpi:
    :param save_format:
    :return:
    """
    data_list = _plotting_input_validator(data_list)

    for dataset in data_list:
        if num_bins is None:
            # specify the bin width
            bin_width = 0.01

            # calculate the number of bins
            dataset = np.array(dataset)
            num_bins = int(np.ceil((dataset.max() - dataset.min()) / bin_width))

        # plot histogram
        plt.hist(dataset, bins=num_bins, color=colors[0], alpha=alpha, **kwargs)

    if label is not None:
        plt.legend(label, loc='upper left')
    if title is not None:
        plt.title(title)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlabel is not None:
        plt.xlabel(xlabel)
    else:
        plt.ylabel("Frequency")
    if save_plot:
        if save_path is None:
            save_path = os.getcwd() + f'fig'
        plt.savefig(fname=save_path, bbox_inches='tight', dpi=dpi, format=save_format)
    if show_plot:
        plt.show()


def plot_matched_data(data_list: list, labels: list,
                      figsize=(10, 10), line_color='lightgrey', linewidth=0.5,
                      marker_colors=['black', 'black'], marker='o',
                      show_boxplot=True, plot_rectangle=False, rectangle_index=None, rectangle_colors=None,
                      xlabel=None, ylabel=None, ylim=None, yticks=None,
                      show_plot=True, save_plot=False, save_path=None, dpi=300, save_format='png'):
    """
    Plots two samples of matched data. Each sample will be plotted as points stacked vertically within condition.
    Lines will be drawn to connect the matching pairs.

    :param save_format:
    :param yticks:
    :param rectangle_index:
    :param ylim:
    :param show_boxplot:
    :param marker:
    :param linewidth:
    :param line_color:
    :param figsize:
    :param data_list:
    :param labels: list of str containing the labels for sample_1 and sample_2
    :param marker_colors: list of str containing matplotlib color styles
    :param xlabel: str
    :param ylabel: str
    :param show_plot: bool
    :param save_plot: bool
    :param save_path: str containing the file path for saving the plot
    :param dpi: int
    :param plot_rectangle: bool indicating whether to plot a rectangle over locs2
    :param rectangle_colors: str indicating the color of the rectangle
    """
    # Check input
    data = _plotting_input_validator(data_list)
    if len(data) != 2:
        ValueError(
            'plot_matched_data() supports analysis of two matched datasets. Length of data_list parameter must equal 2.')

    # Put into dataframe
    df = pd.DataFrame({labels[0]: data[0], labels[1]: data[1]})
    data = pd.melt(df)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.swarmplot(data=data, x='variable', y='value', ax=ax, size=0)
    idx0 = 0
    idx1 = 1
    locs1 = ax.get_children()[idx0].get_offsets()
    locs2 = ax.get_children()[idx1].get_offsets()

    for i in range(locs1.shape[0]):
        x = [locs1[i, 0], locs2[i, 0]]
        y = [locs1[i, 1], locs2[i, 1]]
        ax.plot(x, y, color=line_color, linewidth=linewidth)
        ax.plot(locs1[i, 0], locs1[i, 1], marker=marker, color=marker_colors[0])
        ax.plot(locs2[i, 0], locs2[i, 1], marker=marker, color=marker_colors[1])
        data = [locs1[:, 1], locs2[:, 1]]
        if show_boxplot:
            box_color = 'k'
            sym_color = 'k'
        else:
            box_color = (0, 0, 0, 0)
            sym_color = 'none'
        ax.boxplot(data, positions=[0, 1], sym=sym_color, capprops=dict(linewidth=0.5, color=box_color),
                   whiskerprops=dict(linewidth=0.5, color=box_color),
                   boxprops=dict(linewidth=0.5, color=box_color, facecolor=(0, 0, 0, 0)),
                   medianprops=dict(color=box_color), patch_artist=True)

    # Create a rectangle patch
    if plot_rectangle:
        if ylim is None:
            # Todo: find max value, do not simply set to 1
            ylim = (0, 1)
        if rectangle_index == 1:
            # Add rectangle to rightmost boxplot
            start_x = 0.6
            start_y = ylim[0]
            width_x = 0.8
            width_y = ylim[1] - ylim[0]
            rect = patches.Rectangle((start_x, start_y), width_x, width_y, linewidth=1, edgecolor='none',
                                     facecolor=rectangle_colors[0], alpha=0.2)
            ax.add_patch(rect)
        elif rectangle_index == 0:
            # Add rectangle to leftmost data
            start_x = -0.4
            start_y = ylim[0]
            width_x = 0.8
            width_y = ylim[1] - ylim[0]
            rect = patches.Rectangle((start_x, start_y), width_x, width_y, linewidth=1, edgecolor='none',
                                     facecolor=rectangle_colors[0], alpha=0.2)
            ax.add_patch(rect)
        else:
            left_rect = patches.Rectangle((-0.4, ylim[0]), 0.8, ylim[1] - ylim[0], linewidth=1, edgecolor='none',
                                          facecolor=rectangle_colors[0], alpha=0.2)
            right_rect = patches.Rectangle((0.6, ylim[0]), 0.8, ylim[1] - ylim[0], linewidth=1, edgecolor='none',
                                           facecolor=rectangle_colors[1], alpha=0.2)
            ax.add_patch(left_rect)
            ax.add_patch(right_rect)

        # Add the rectangle patch to the plot

    if yticks is not None:
        plt.yticks(yticks)
    plt.xticks([])
    if ylim is not None:
        plt.ylim(ylim)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlabel is None:
        plt.xlabel(f'P-value = {stats.ttest_rel(data[0], data[1]).pvalue:.3}')
    sns.despine(offset=10, trim=True)
    if save_plot:
        if save_path is None:
            save_path = os.getcwd() + f'fig'
        plt.savefig(fname=save_path, bbox_inches='tight', dpi=dpi, format=save_format)
    if show_plot:
        plt.show()


def plot_heatmap(data_matrix, title=None, show_plot=True, save_plot=False, save_path=None, dpi=300, save_format='png',
                 **kwargs):
    sns.heatmap(data=data_matrix, xticklabels=False, yticklabels=False, **kwargs)
    if title is not None:
        plt.title(title)
    if save_plot:
        if save_path is None:
            save_path = os.getcwd() + f'fig'
        plt.savefig(fname=save_path, bbox_inches='tight', dpi=dpi, format=save_format)
    if show_plot:
        plt.show()
