# CaGraph imports
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pynwb import NWBHDF5IO
from statsmodels.tsa.stattools import grangercausalitytests
from scipy import stats

# Visualization imports
from bokeh.io import show, save
from bokeh.models import Range1d, Circle, MultiLine
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.palettes import Blues8
from bokeh.transform import linear_cmap
import scipy
import pandas as pd
import os

# Benchmarking imports

#%% CaGraph class

class CaGraph:
    """
    Published: XX/XX/XXXX
    Author: Veronica Porubsky [Github: https://github.com/vporubsky][ORCID: https://orcid.org/0000-0001-7216-3368]

    Class: CaGraph(data_file, identifiers=None)
    =====================

    This class provides functionality to easily visualize time-series data of
    neuronal activity and to compute correlation metrics of neuronal networks,
    and generate graph objects which can be analyzed using graph theory.
    There are several graph theoretical metrics for further analysis of
    neuronal network connectivity patterns.

    Most test_analyses are computed using a graph generated based on Pearson's correlation coefficient values computed
    between neuron timeseries data. A threshold for Pearson's correlation coefficient is typically set edges added
    when R>0.3. https://www.nature.com/articles/s41467-020-17270-w#MOESM1 https://www.nature.com/articles/nature15389
    ...

    Attributes
    ----------
    data_file : str
        A string pointing to the file to be used for data analysis.
    identifiers : list
        A list of identifiers for each row of calcium imaging data in
        the data_file passed to CaGraph.

    Methods
    -------
    get_laplacian_matrix()
        ...
    Todo: Add additional graph theoretical test_analyses (path length, rich club, motifs...)
    Todo: Add additional correlation metrics and allow user to pass them (currently only Pearson)
    Todo: Determine the distribution of eigenvector centrality scores in connected modules/subnetworks
    Todo: Implement shuffle distribution r value correction: https://www.nature.com/articles/s41467-020-17270-w#MOESM1
    Todo: Add threshold setting utilties which evaluate whether a given dataset is a good candidate
     --- Eventually you will need to add to this so it includes event detection --- ask for this code

    """

    def __init__(self, data_file, labels=None, dataset_id=None, threshold=None):
        """

        :param csv_file: str
        :param identifiers:
        :param dataset_id: str
        """
        if isinstance(data_file, np.ndarray):
            self.data = data_file
            self.time = self.data[0, :]
            self.neuron_dynamics = self.data[1:len(self.data), :]
        elif data_file.endswith('csv'):
            self.data = np.genfromtxt(data_file, delimiter=",")
            self.time = self.data[0, :]
            self.neuron_dynamics = self.data[1:len(self.data), :]
        elif data_file.endswith('nwb'):
            with NWBHDF5IO(data_file, 'r') as io:
                nwbfile_read = io.read()
                nwb_acquisition_key = list(nwbfile_read.acquisition.keys())[0]
                ca_from_nwb = nwbfile_read.acquisition[nwb_acquisition_key]
                self.neuron_dynamics = ca_from_nwb.data[:]
                self.time = ca_from_nwb.timestamps[:]
        else:
            print('Data must be passed as a .csv or .nwb file.')
            raise TypeError
        if dataset_id is not None:
            self.data_id = dataset_id
        self.data_filename = str(data_file)
        self.time = self.data[0, :]
        self.dt = self.time[1] - self.time[0]
        self.neuron_dynamics = self.data[1:len(self.data), :]
        self.num_neurons = np.shape(self.neuron_dynamics)[0]
        if labels is None:
            self.labels = np.linspace(0, np.shape(self.neuron_dynamics)[0] - 1, \
                                      np.shape(self.neuron_dynamics)[0]).astype(int)
        else:
            self.labels = labels
        self.pearsons_correlation_matrix = np.nan_to_num(np.corrcoef(self.neuron_dynamics))
        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = self.__generate_threshold()

    def __generate_threshold(self):
        """

        """
        return 0.3

    def get_laplacian_matrix(self, graph=None, threshold=0.3):
        """
        Returns the Laplacian matrix of the specified graph.

        :param graph:
        :param threshold:
        :return:
        """
        if graph is None:
            graph = self.get_network_graph_from_matrix(threshold=threshold)
        return nx.laplacian_matrix(graph)

    def get_network_graph_from_matrix(self, threshold=0.3, weight_matrix=None):
        """
        Automatically generate graph object from numpy adjacency matrix.

        :param weight_matrix:
        :param threshold:
        :param weighted:
        :return:
        """
        if weight_matrix is None:
            return nx.from_numpy_array(self.get_adjacency_matrix(threshold=threshold))
        return nx.from_numpy_array(self.get_weight_matrix(weight_matrix=weight_matrix))

    def get_pearsons_correlation_matrix(self, data_matrix=None, time_points=None):
        """
        Returns the Pearson's correlation for all neuron pairs.

        :param data_matrix:
        :param time_points: tuple
        :return:
        """
        if data_matrix is None:
            data_matrix = self.neuron_dynamics
        if time_points:
            data_matrix = data_matrix[:, time_points[0]:time_points[1]]
        return np.nan_to_num(np.corrcoef(data_matrix, rowvar=True))

    def get_time_subsampled_graphs(self, subsample_indices, threshold=0.3, weighted=False):
        """

        :param subsample_indices: list of tuples
        :param threshold:
        :return:
        """
        subsampled_graphs = []
        for time_idx in subsample_indices:
            subsampled_graphs.append(
                self.get_network_graph(corr_mat=self.get_pearsons_correlation_matrix(time_points=time_idx),
                                       threshold=threshold, weighted=weighted))
        return subsampled_graphs

    def get_time_subsampled_correlation_matrix(self, subsample_indices, threshold=0.3):
        """

        :param subsample_indices: list of tuples
        :param threshold:
        :return:
        """
        subsampled_corr_mat = []
        for time_idx in subsample_indices:
            subsampled_corr_mat.append(self.get_pearsons_correlation_matrix(time_points=time_idx))
        return subsampled_corr_mat

    def get_granger_causality_scores_matrix(self):
        """

        :return:
        """
        r, c = np.shape(self.neuron_dynamics)
        gc_matrix = np.zeros((r, r))
        for row in range(r):
            for col in range(r):
                gc_test_dict = grangercausalitytests(np.transpose(self.neuron_dynamics[[row, col], :]), \
                                                     maxlag=1, verbose=False)[1][0]
                gc_matrix[row, col] = gc_test_dict['ssr_chi2test'][1]
        return gc_matrix

    def get_adjacency_matrix(self, threshold=0.3):
        """

        :param threshold:
        :return:
        """
        adj_mat = (self.pearsons_correlation_matrix > threshold)
        np.fill_diagonal(adj_mat, 0)
        return adj_mat.astype(int)

    def get_weight_matrix(self, weight_matrix=None):
        """Returns a weighted connectivity matrix with zero along the diagonal. No threshold is applied.
        :param weight_matrix: numpy.ndarray containing weights
        :return:
        """
        if weight_matrix is None:
            weight_matrix = self.pearsons_correlation_matrix
        np.fill_diagonal(weight_matrix, 0)
        return weight_matrix

    def plot_correlation_heatmap(self, correlation_matrix=None):
        """

        :param correlation_matrix:
        :return:
        """
        if correlation_matrix is None:
            correlation_matrix = self.get_pearsons_correlation_matrix()
        sns.heatmap(correlation_matrix, vmin=0, vmax=1)
        return

    def get_single_neuron_timecourse(self, neuron_trace_number):
        """
        Return time vector stacked on the recorded neuron of interest.

        :param neuron_trace_number:
        :return:
        """
        neuron_timecourse_selection = neuron_trace_number
        return np.vstack((self.time, self.neuron_dynamics[neuron_timecourse_selection, :]))

    # todo: make units flexible/ allow user to pass plotting information
    def plot_single_neuron_timecourse(self, neuron_trace_number, title=None):
        """

        :param neuron_trace_number:
        :return:
        """
        neuron_timecourse_selection = neuron_trace_number
        count = 1
        x_tick_array = []
        for i in range(len(self.time)):
            if count % (len(self.time) / 20) == 0:
                x_tick_array.append(self.time[i])
            count += 1
        plt.figure(num=1, figsize=(10, 2))
        plt.plot(self.time, self.neuron_dynamics[neuron_timecourse_selection, :],
                 linewidth=1)  # add option : 'xkcd:olive',
        plt.xticks(x_tick_array)
        plt.xlim(0, self.time[-1])
        plt.ylabel('ΔF/F')
        plt.xlabel('Time (s)')
        if title is None:
            plt.title(f'{self.data_id} neuron {neuron_timecourse_selection}')
        else:
            plt.title(title)
        plt.show()

    def plot_multi_neuron_timecourse(self, neuron_trace_numbers, title=None, palette=None, show=False):
        """
        Plots multiple individual calcium fluorescence traces, stacked vertically.

        :param graph:
        :param threshold:
        :param title:
        :return:
        """
        count = 0
        if palette is None:
            palette = sns.color_palette('husl', len(neuron_trace_numbers))
        for idx, neuron in enumerate(neuron_trace_numbers):
            y = self.neuron_dynamics[neuron, :].copy() / max(self.neuron_dynamics[neuron, :])
            y = [x + 1.05 * count for x in y]
            plt.plot(self.time, y, c=palette[idx], linewidth=1)
            plt.xticks([])
            plt.yticks([])
            count += 1
        plt.ylabel('ΔF/F')
        plt.xlabel('Time (s)')
        if title: plt.title(title)
        if show: plt.show()

    # Todo: plot stacked timecourses based on input neuron indices from graph theory test_analyses
    # Todo: adjust y axis title for normalization
    # Todo: add time ticks
    # Todo: check that the self.num_neurons is not too many for color_palette
    def plot_subnetworks_timecourses(self, graph=None, threshold=0.3, palette=None, title=None):
        """

        :param graph:
        :param threshold:
        :param title:
        :return:
        """
        subnetworks = self.get_subnetworks(graph=graph, threshold=threshold)
        if palette is None:
            palette = sns.color_palette('husl', self.num_neurons)
        for idx, subnetwork in enumerate(subnetworks):
            count = 0
            for neuron in subnetwork:
                y = self.neuron_dynamics[neuron, :].copy() / max(self.neuron_dynamics[neuron, :])
                for j in range(len(y)):
                    y[j] = y[j] + 1.05 * count
                plt.plot(self.time, y, c=palette[idx], linewidth=1)
                plt.xticks([])
                plt.yticks([])
                count += 1
            plt.ylabel('ΔF/F')
            plt.xlabel('Time (s)')
            if title: plt.title(title)
            plt.show()

    def plot_multi_neurons_timecourses(self, graph=None, threshold=0.3, title=None):
        """

        :param graph:
        :param threshold:
        :param title:
        :return:
        """
        subnetworks = self.get_subnetworks(graph=graph, threshold=threshold)
        for subnetwork in subnetworks:
            count = 0
            for neuron in subnetwork:
                y = self.neuron_dynamics[neuron, :].copy() / max(self.neuron_dynamics[neuron, :])
                for j in range(len(y)):
                    y[j] = y[j] + 1.05 * count
                plt.plot(self.time, y, 'k', linewidth=1)
                plt.xticks([])
                plt.yticks([])
                count += 1
            plt.ylabel('ΔF/F')
            plt.xlabel('Time (s)')
            if title: plt.title(title)
            plt.show()

    def plot_all_neurons_timecourse(self):
        """

        """
        plt.figure(num=2, figsize=(10, 2))
        count = 1
        x_tick_array = []
        for i in range(len(self.time)):
            if count % (len(self.time) / 20) == 0:
                x_tick_array.append(self.time[i])
            count += 1
        for i in range(len(self.neuron_dynamics) - 1):
            plt.plot(self.time, self.neuron_dynamics[i, :], linewidth=0.5)
            plt.xticks(x_tick_array)
            plt.xlim(0, self.time[-1])
            plt.ylabel('ΔF/F')
            plt.xlabel('Time (s)')
            plt.title(f'{self.data_id}')
        plt.show()

    # Todo: ensure the binary and weighted graphs are built correctly
    # Todo: ensure this is the most efficient graph building method
    def get_network_graph(self, corr_mat=None, threshold=0.3, weighted=False):
        """
        Must pass a np.ndarray type object to corr_mat, or the Pearsons
        correlation matrix for the full dataset will be used.

        :param corr_mat:
        :param threshold:
        :param weighted:
        :return:
        """
        if not isinstance(corr_mat, np.ndarray):
            corr_mat = self.pearsons_correlation_matrix  # update to include other correlation metrics
        G = nx.Graph()
        if weighted:
            for i in range(len(self.labels)):
                G.add_node(str(self.labels[i]))
                for j in range(len(self.labels)):
                    if not i == j:
                        G.add_edge(str(self.labels[i]), str(self.labels[j]), weight=corr_mat[i, j])
        else:
            for i in range(len(self.labels)):
                G.add_node(str(self.labels[i]))
                for j in range(len(self.labels)):
                    if not i == j and corr_mat[i, j] > threshold:
                        G.add_edge(str(self.labels[i]), str(self.labels[j]))
        return G

    # Todo: update to be able to pass a graph for randomization
    def get_random_graph(self, threshold=0.3):
        """
        nx.algorithms.smallworld.random_reference is adapted from the Maslov and Sneppen (2002) algorithm.
        It randomizes the existing graph.

        :param threshold:
        :return:
        """
        G = self.get_network_graph(threshold=threshold)
        G = nx.algorithms.smallworld.random_reference(G)
        return G

    def get_erdos_renyi_graph(self, graph=None, threshold=0.3):
        """
        Generates an Erdos-Renyi random graph using a network edge coverage
        metric computed from the graph to be randomized.

        :param graph:
        :param threshold:
        :return:
        """
        if graph is None:
            num_nodes = self.num_neurons
            con_probability = self.get_network_coverage(threshold=threshold)
        else:
            num_nodes = len(graph.nodes)
            con_probability = self.get_network_coverage(graph=graph)
        return nx.erdos_renyi_graph(n=num_nodes, p=con_probability)

    def plot_graph_network(self, graph, position=None):
        """

        :param graph:
        :param position:
        :return:
        """
        if not position:
            position = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos=position, node_color='b', node_size=100)
        nx.draw_networkx_edges(graph, pos=position, edge_color='b', )
        nx.draw_networkx_labels(graph, pos=position, font_size=6, font_color='w', font_family='sans-serif')
        plt.axis('off')
        plt.show()
        return

    # Todo: getting stuck on small world analysis when computing sigma -- infinite loop
    # Todo: this^ may be due to computing the average clustering coefficient or the average shortest path length -- test
    def get_smallworld_largest_subnetwork(self, graph=None, threshold=0.3):
        """

        :param graph:
        :param threshold:
        :return:
        """
        if graph is None:
            graph = self.get_largest_subnetwork_graph(threshold=threshold)
        else:
            graph = self.get_largest_subnetwork_graph(graph=graph, threshold=threshold)
        if len(graph.nodes()) >= 4:
            return nx.algorithms.smallworld.sigma(graph)
        else:
            raise RuntimeError(
                'Largest subnetwork has less than four nodes. networkx.algorithms.smallworld.sigma cannot be computed.')

    # Todo: DO NOT USE FUNCTION, NOT UPDATED
    def get_smallworld_all_subnetworks(self, corr_matrix, graph=None, threshold=0.3):
        """

        :param corr_matrix:
        :param G:
        :param threshold:
        :return:
        """
        if graph is None:
            graph = self.get_network_graph(corr_matrix, threshold)
        graph_max_subgraph_generator = sorted(nx.connected_components(graph), key=len, reverse=True)
        omega_list = []
        for i, val in enumerate(graph_max_subgraph_generator):
            graph_max_subgraph = graph.subgraph(val)
            if len(graph_max_subgraph.nodes) >= 4:
                omega = nx.algorithms.smallworld.omega(graph_max_subgraph)
                omega_list.append(omega)
        return omega_list

    # Todo: define hits_threshold based on tail of powerlaw distribution
    # Todo: determine best practices for setting threshold of powerlaw distribution to find hubs
    def get_hubs(self, graph=None, threshold=0.3):
        """

        :param graph:
        :param threshold:
        :return:
        """
        if graph is None:
            hubs, authorities = nx.hits(self.get_network_graph(threshold=threshold))
        else:
            hubs, authorities = nx.hits(graph)
        med_hubs = np.median(list(hubs.values()))
        std_hubs = np.std(list(hubs.values()))
        hubs_threshold = med_hubs + 2.5 * std_hubs
        hubs_list = []
        [hubs_list.append(x) for x in hubs.keys() if hubs[x] > hubs_threshold]
        return hubs_list, hubs

    def get_subnetworks(self, graph=None, threshold=0.3):
        """

        :param graph:
        :param threshold:
        :return:
        """
        if graph is None:
            connected_components = list(nx.connected_components(self.get_network_graph(threshold=threshold)))
        else:
            connected_components = list(nx.connected_components(graph))
        subnetworks = []
        [subnetworks.append(list(map(int, x))) for x in connected_components if len(x) > 1]
        return subnetworks

    def get_largest_subnetwork_graph(self, graph=None, threshold=0.3):
        """

        :param graph:
        :param threshold:
        :return:
        """
        if graph is None:
            graph = self.get_network_graph(threshold=threshold)
        largest_component = max(nx.connected_components(graph), key=len)
        return graph.subgraph(largest_component)

    # Todo: add functionality get_path_length
    def get_path_length(self):
        """
        Returns the characteristic path length.

        :return:
        """
        return

    def get_clustering_coefficient(self, threshold=0.3, graph=None):
        """

        :param threshold:
        :param G:
        :return:
        """
        if graph is None:
            graph = self.get_network_graph_from_matrix(threshold=threshold)
        degree_view = nx.clustering(graph)
        clustering_coefficient = []
        [clustering_coefficient.append(degree_view[node]) for node in graph.nodes()]
        return clustering_coefficient

    def get_degree(self, graph=None, threshold=0.3):
        """
        Returns iterator object of (node, degree) pairs.

        :param threshold:
        :return:
        """
        if graph is None:
            return self.get_network_graph_from_matrix(threshold=threshold)
        else:
            return graph.degree

    # Todo: note that description is from https://www.nature.com/articles/s41467-020-17270-w#Sec8
    def get_correlated_pair_ratio(self, graph=None, threshold=0.3, ):
        """
        Computes the number of connections each neuron has, divided by the nuber of cells in the field of view.

        :param graph:
        :param threshold:
        :param G:
        :return:
        """
        if graph is None:
            graph = self.get_network_graph_from_matrix(threshold=threshold)
        degree_view = self.get_degree(graph)
        correlated_pair_ratio = []
        [correlated_pair_ratio.append(degree_view[node] / self.num_neurons) for node in graph.nodes()]
        return correlated_pair_ratio

    # Todo: adapt for directed - current total possible edges is for undirected
    def get_network_coverage(self, graph=None, threshold=0.3):
        """
        Returns the percentage of edges present in the network
        out of the total possible edges.

        :param graph:
        :param threshold:
        :return:
        """
        possible_edges = (self.num_neurons * (self.num_neurons - 1)) / 2
        if graph is None:
            graph = self.get_network_graph(threshold=threshold)
        return len(graph.edges) / possible_edges

    # Todo: check form of centrality
    def get_eigenvector_centrality(self, graph=None, threshold=0.3):
        """
        Compute the eigenvector centrality of all network nodes, the
        measure of influence each node has on the network.

        :param graph:
        :param threshold:
        :return:
        """
        if graph is None:
            graph = self.get_network_graph_from_matrix(threshold=threshold)
        centrality = nx.eigenvector_centrality(graph)
        return centrality

    def get_communities(self, graph=None, threshold=0.3):
        """

        :param graph:
        :param threshold:
        :return: node_groups:
        """
        if graph is None:
            graph = self.get_network_graph_from_matrix(threshold=threshold)
        communities = nx.algorithms.community.centrality.girvan_newman(graph)
        node_groups = []
        for community in next(communities):
            node_groups.append(list(community))
        return node_groups

    # Todo: add additional arguments
    def draw_network(self, graph=None, node_size=25, node_color='b', alpha=0.5):
        """

        :param graph:
        :param node_size:
        :param node_color:
        :param alpha:
        :return:
        """
        if graph is None:
            graph = self.get_network_graph()
        nx.draw(graph, pos=nx.spring_layout(graph), node_size=node_size, node_color=node_color, alpha=alpha)


#%% Functionality to preprocess the dataset and validate the choice of parameters.

class Visualization:
    """
    Published: XX/XX/XXXX
    Author: Veronica Porubsky [Github: https://github.com/vporubsky][ORCID: https://orcid.org/0000-0001-7216-3368]

    Class: Visualization()
    =====================

    This class provides functionality to easily visualize graphs computed using the CaGraph class.


    Attributes
    ----------


    """
    def __init__(self):
        pass

    def interactive_network(self, ca_graph_obj, graph=None, attributes=['degree', 'HITS', 'hubs', 'CPR', 'communities'],
                            adjust_node_size=5, adjust_size_by='degree', adjust_color_by='communities',
                            palette=Blues8,
                            hover_attributes=['degree', 'HITS', 'hubs', 'CPR', 'communities'], title=None,
                            show_plot=True, save_plot=False, save_path=None):
        """
        Generates an interactived Bokeh.io plot of the graph network.

        palette: a color palette which can be passed as a tuple: palette = ('grey', 'red', 'blue')

        """
        # initialize graph information
        cg = ca_graph_obj
        if graph is None:
            G = cg.get_network_graph()
        else:
            G = graph
        label_keys = list(map(str, list(cg.labels)))

        ## Todo: add additional attributes
        #  Build attributes dictionary
        attribute_dict = {}
        for attribute in attributes:
            if attribute == 'degree':
                # Compute the degree of each node and add attribute
                attribute_dict['degree'] = dict(nx.degree(G))
            elif attribute == 'HITS':
                # Add HITS attribute
                hub_list, hit_vals = cg.get_hubs(graph=G)
                attribute_dict['HITS'] = hit_vals
            elif attribute == 'hubs':
                # Add hubs attribute
                attribute_dict['hubs'] = {i: 1 if i in list(set(hub_list) & set(label_keys)) else 0 for i in
                                          label_keys}
            elif attribute == 'CPR':
                # Add correlated pairs attribute
                corr_pair = cg.get_correlated_pair_ratio(graph=G)
                attribute_dict['CPR'] = {i: j for i, j in zip(label_keys, corr_pair)}
            elif attribute == 'communities':
                # Add communities
                c = list(nx.algorithms.community.greedy_modularity_communities(G))
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
            nx.set_node_attributes(G, name=key, values=value)

        # Todo: make this flexible for labeled conditions
        # # Add context active attribute
        # con_act = list(np.genfromtxt(mouse_id + '_neuron_context_active.csv', delimiter=','))  # load context active designations
        # con_act_dict: dict = {i: j for i, j in zip(label_keys, con_act)}
        # networkx.set_node_attributes(G, name='context_activity', values=con_act_dict)

        # Adjusted node size
        if adjust_node_size is not None:
            # Adjust node size
            adjusted_node_size = dict(
                [(node, value + adjust_node_size) for node, value in attribute_dict[adjust_size_by].items()])
            nx.set_node_attributes(G, name='adjusted_node_size', values=adjusted_node_size)
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
        if show_plot:
            show(plot)

        if save_plot:
            if save_path is not None:
                save(plot, filename=save_path)
            else:
                save(plot, filename=os.path.join(os.getcwd(), f"bokeh_graph_visualization.html"))

    def plot_CDF(self, data=None, color='black', marker='o', x_label='', y_label='CDF', show_plot=False):
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

    def plot_CDF_compare_two_samples(self, data_list=None, color_list=['black', 'black'], marker='o', x_label='',
                                     y_label='CDF', show_plot=False):

        """
        Plots the cumulative distribution function of the provided datasets and prints the associated P-value for assessing
        the Kolmogorov-Smirnov distance between the distributions.

        :param data_list: list of lists containing float values to compare with KS-test
        :param color_list: list of str containing matplotlib color styles
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
            plt.plot(sorted_data, cdf, color=color_list[idx], marker=marker)

        plt.ylabel(y_label)
        plt.title(f'P value: {stat_level.pvalue:.2e}')

        if show_plot:
            plt.show()
        plt.xlabel(x_label)

    # Todo: check functionality
    def plot_matched_data(self, sample_1, sample_2, labels, colors):
        """
        Plots two samples of matched data with each sample

        """
        # Put into dataframe
        df = pd.DataFrame({labels[0]: sample_1, labels[1]: sample_2})
        data = pd.melt(df)

        # Plot
        fig, ax = plt.subplots()
        sns.swarmplot(data=data, x='variable', y='value', ax=ax, size=0)

        # Find idx0 and idx1 by inspecting the elements returned from ax.get_children()
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

        plt.xlabel(f'P-value = {scipy.stats.ttest_rel(sample_1, sample_2).pvalue:.3}')


#%% Functionality to preprocess the dataset and validate the choice of parameters.

class Preprocess:
    def __init__(self):
        pass

    # ---------------- Clean data --------------------------------------
    # Todo: look up smoothing algorithm for calcium data
    def smooth(self, data):
        """
        Smooth unprocessed data to remove noise.
        """
        smoothed_data = data
        return smoothed_data

    # Todo: make auto-clean option for those that don't have experience
    def auto_preprocess(self, data):
        """

        """
        preprocessed_data = self.smooth(data)
        return preprocessed_data

    # Todo: write event_detection code
    def event_detection(self, data):
        """

        :param data:
        :return:
        """
        event_data = data
        return event_data

    # Todo: create function to generate event_data
    def remove_quiescent(self, data, event_data, event_num_threshold=5):
        """
        data: numpy.ndarray
        event_bins: numpy.ndarray

        Removes inactive neurons from the dataset using event_data which the user must pass.
        """
        binarized_event_data = np.where(event_data > 0.0005, 1, 0)
        new_event_data = np.zeros((1, np.shape(event_data)[1]))
        new_data = np.zeros((1, np.shape(data)[1]))
        for row in range(np.shape(binarized_event_data)[0]):
            if np.sum(binarized_event_data[row, :]) <= event_num_threshold:
                continue
            else:
                new_event_data = np.vstack((new_event_data, event_data[row,:]))
                new_data = np.vstack((new_data, data[row,:]))
        return new_data[1:, :], new_event_data[1:,:]

    def __count_sign_switch(self, row_data):
        """

        """
        subtract = row_data[0:len(row_data)-1] - row_data[1:]
        a = subtract
        asign = np.sign(a)
        signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
        return np.sum(signchange)

    def remove_low_activity(self, data, event_data, event_num_threshold=5):
        """
        Removes neurons who have fewer than event_num_threshold events.

        Returns a new array of data without neurons that have low activity.
        """
        #apply activity treshold
        new_event_data = np.zeros((1, np.shape(event_data)[1]))
        new_data = np.zeros((1, np.shape(data)[1]))
        for row in range(np.shape(data)[0]):
            if self.__count_sign_switch(row_data = data[row,:]) <= 5 and not row == 0:
                continue
            else:
                new_event_data = np.vstack((new_event_data, event_data[row,:]))
                new_data = np.vstack((new_data, data[row,:]))
        return new_data[1:, :], new_event_data[1:,:]

    # Suitability for graph theory analysis
    def __bins(self, lst, n):
        """
        Yield successive n-sized chunks from lst.
        """
        lst = list(lst)
        build_binned_list = []
        for i in range(0, len(lst), n):
            build_binned_list.append(lst[i:i + n])
        return build_binned_list

    def generate_randomized_timeseries_matrix(self, data: list) -> np.ndarray:
        """
        data: list

        Parameter data should contain a list of np.ndarray objects.

        Return a numpy array or NWB file.
        """
        time = data[0, :].copy()
        for row in range(np.shape(data)[0]):
            np.random.shuffle(data[row, :])
        data[0, :] = time.copy()
        return data

    def generate_randomized_timeseries_binned(self, data: list, bin_size: int) -> np.ndarray:
        """
        data: list

        Parameter data should contain a list of np.ndarray objects.

        Return a numpy array or NWB file.
        """
        time = data[0, :].copy()
        build_new_array = np.array(self.__bins(lst=data[1, :], n=bin_size))

        # build binned dist
        for row in range(np.shape(data[2:, :])[0]):
            binned_row = self.__bins(lst=data[row + 2, :], n=bin_size)
            build_new_array = np.vstack([build_new_array, binned_row])

        for row in range(np.shape(build_new_array)[0]):
            np.random.shuffle(build_new_array[row, :])

        flatten_array = time.copy()
        for row in range(np.shape(build_new_array)[0]):
            flat_row = [item for sublist in build_new_array[row, :] for item in sublist]
            flatten_array = np.vstack([flatten_array, flat_row])

        return flatten_array

    def __event_bins(self, data, events):
        """
        :param data:
        :param events: single events timecourse
        :return:
        """
        data = list(data)
        build_binned_list = []
        event_idx = list(np.nonzero(events)[0])
        if event_idx[-1] != len(data):
            event_idx.append(len(data))
        start_val = 0
        for idx in event_idx:
            build_binned_list.append(data[start_val:idx])
            start_val = idx
        np.random.shuffle(build_binned_list)
        flat_random_binned_list = [item for sublist in build_binned_list for item in sublist]
        threshold = 0.01
        flat_random_binned_list = [0 if value < threshold else value for value in flat_random_binned_list]
        return flat_random_binned_list

    def generate_event_segmented(self, data: list, event_data: list) -> np.ndarray:
        """
        data: list

        Parameter data should contain a list of np.ndarray objects.

        Return a numpy array or NWB file.
        """
        time = data[0, :].copy()

        # build binned dist
        flatten_array = time.copy()
        for row in range(np.shape(data[1:, :])[0]):
            binned_row = self.__event_bins(data=data[row + 1, :], events=event_data[row + 1, :])
            flatten_array = np.vstack([flatten_array, binned_row])

        return flatten_array

    def generate_randomized(self, data: list, bin_size: int) -> np.ndarray:
        """
        data: list

        Parameter data should contain a list of np.ndarray objects.

        Return a numpy array or NWB file.
        """
        time = data[0, :].copy()

        # build binned dist
        flatten_array = time.copy()
        for row in range(np.shape(data[2:, :])[0]):
            binned_row = self.__bins(lst=data[row + 2, :], n=bin_size)
            flatten_array = np.vstack([flatten_array, binned_row])

        return flatten_array

    #  Todo: write function
    def generate_randomized_across_population(self, data: np.ndarray, event_data: np.ndarray) -> np.ndarray:

        # First split all the data and make a long list of
        return


    # Todo: function to find outlier subjects in a batch



    # Todo: function to test sensitivity analysis
    def sensitivity_analysis(self):

        return

