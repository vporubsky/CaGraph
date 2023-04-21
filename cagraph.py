# CaGraph imports
import preprocess as prep
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pynwb import NWBHDF5IO
from statsmodels.tsa.stattools import grangercausalitytests


# %% CaGraph class

class CaGraph:
    """
    Author: Veronica Porubsky [Github: https://github.com/vporubsky][ORCID: https://orcid.org/0000-0001-7216-3368]

    Class: CaGraph(data_file, identifiers=None)
    =====================

    This class provides functionality to easily visualize time-series data of
    neuronal activity and to compute correlation metrics of neuronal networks,
    and generate graph objects which can be analyzed using graph theory.
    There are several graph theoretical metrics for further analysis of
    neuronal network connectivity patterns.


    Attributes
    ----------
    data_file : str
        A string pointing to the file to be used for data analysis.
    labels: list
        A list of identifiers for each row of calcium imaging data in
        the data_file passed to CaGraph.

    Methods
    -------
    get_laplacian_matrix()
        ...


    """

    def __init__(self, data_file, labels=None, dataset_id=None, threshold=None):
        """

        :param data_file: str
        :param labels:
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
            print('Data must be passed as a .csv or .nwb file, or as numpy.ndarray.')
            raise TypeError
        if dataset_id is not None:
            self.data_id = dataset_id
        self.data_filename = str(data_file)
        self.time = self.data[0, :]
        self.dt = self.time[1] - self.time[0]
        self.neuron_dynamics = self.data[1:len(self.data), :]
        self.num_neurons = np.shape(self.neuron_dynamics)[0]
        if labels is None:
            self.labels = np.linspace(0, np.shape(self.neuron_dynamics)[0] - 1,
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
        Generates a threshold for the provided datatset as described in the preprocess module.

        :return: float
        """
        return prep.generate_threshold(data=self.neuron_dynamics)


    def get_laplacian_matrix(self, graph=None):
        """
        Returns the Laplacian matrix of the specified graph.

        :param graph:
        :return:
        """
        if graph is None:
            graph = self.get_graph_from_matrix()
        return nx.laplacian_matrix(graph)

    def get_graph_from_matrix(self, weighted=False):
        """
        Automatically generate graph object from numpy adjacency matrix.

        :param weighted:
        :return:
        """
        if not weighted:
            return nx.from_numpy_array(self.get_adjacency_matrix())
        return nx.from_numpy_array(self.get_weight_matrix())

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

    def get_time_subsampled_graphs(self, subsample_indices, weighted=False):
        """

        :param weighted:
        :param subsample_indices: list of tuples
        :return:
        """
        subsampled_graphs = []
        for time_idx in subsample_indices:
            subsampled_graphs.append(
                self.get_graph(corr_mat=self.get_pearsons_correlation_matrix(time_points=time_idx), weighted=weighted))
        return subsampled_graphs

    def get_time_subsampled_correlation_matrices(self, subsample_indices):
        """
        Samples the timeseries using provided indices to generate correlation matrices for
        defined periods.

        :param subsample_indices: list of tuples
        :return:
        """
        subsampled_corr_mat = []
        for time_idx in subsample_indices:
            subsampled_corr_mat.append(self.get_pearsons_correlation_matrix(time_points=time_idx))
        return subsampled_corr_mat

    def get_granger_causality_scores_matrix(self):
        """
        Returns Granger causality chi-square values.

        :return:
        """
        r, c = np.shape(self.neuron_dynamics)
        gc_matrix = np.zeros((r, r))
        for row in range(r):
            for col in range(r):
                gc_test_dict = grangercausalitytests(np.transpose(self.neuron_dynamics[[row, col], :]),
                                                     maxlag=1, verbose=False)[1][0]
                gc_matrix[row, col] = gc_test_dict['ssr_chi2test'][1]
        return gc_matrix

    def get_adjacency_matrix(self):
        """
        Returns the adjacency matrix.

        :return:
        """
        adj_mat = (self.pearsons_correlation_matrix > self.threshold)
        np.fill_diagonal(adj_mat, 0)
        return adj_mat.astype(int)

    def get_weight_matrix(self):
        """
        Returns a weighted connectivity matrix with zero along the diagonal. No threshold is applied.

        :return:
        """
        weight_matrix = self.pearsons_correlation_matrix
        np.fill_diagonal(weight_matrix, 0)
        return weight_matrix

    def plot_correlation_heatmap(self, correlation_matrix=None, show_plot=True):
        """
        Plots a heatmap of the correlation matrix.

        :param show_plot:
        :param correlation_matrix:
        :return:
        """
        if correlation_matrix is None:
            correlation_matrix = self.get_pearsons_correlation_matrix()
        sns.heatmap(correlation_matrix, vmin=0, vmax=1)
        if show_plot:
            plt.show()
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

        :param title:
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

    def plot_multi_neuron_timecourse(self, neuron_trace_labels, title=None, palette=None, show=False):
        """
        Plots multiple individual calcium fluorescence traces, stacked vertically.

        :param show:
        :param palette:
        :param neuron_trace_labels:
        :param title:
        :return:
        """
        count = 0
        if palette is None:
            palette = sns.color_palette('husl', len(neuron_trace_labels))
        for idx, neuron in enumerate(neuron_trace_labels):
            y = self.neuron_dynamics[neuron, :].copy() / max(self.neuron_dynamics[neuron, :])
            y = [x + 1.05 * count for x in y]
            plt.plot(self.time, y, c=palette[idx], linewidth=1)
            plt.xticks([])
            plt.yticks([])
            count += 1
        plt.ylabel('ΔF/F')
        plt.xlabel('Time (s)')
        if title:
            plt.title(title)
        if show:
            plt.show()

    # Todo: plot stacked timecourses based on input neuron indices from graph theory test_analyses
    # Todo: adjust y axis title for normalization
    # Todo: add time ticks
    # Todo: check that the self.num_neurons is not too many for color_palette
    def plot_subgraphs_timecourses(self, graph=None, palette=None, title=None):
        """

        :param palette:
        :param graph:
        :param title:
        :return:
        """
        subgraphs = self.get_subgraphs(graph=graph)
        if palette is None:
            palette = sns.color_palette('husl', self.num_neurons)
        for idx, subgraph in enumerate(subgraphs):
            count = 0
            for neuron in subgraph:
                y = self.neuron_dynamics[neuron, :].copy() / max(self.neuron_dynamics[neuron, :])
                for j in range(len(y)):
                    y[j] = y[j] + 1.05 * count
                plt.plot(self.time, y, c=palette[idx], linewidth=1)
                plt.xticks([])
                plt.yticks([])
                count += 1
            plt.ylabel('ΔF/F')
            plt.xlabel('Time (s)')
            if title:
                plt.title(title)
            plt.show()

    def plot_multi_neurons_timecourses(self, graph=None, title=None):
        """

        :param graph:
        :param title:
        :return:
        """
        subgraphs = self.get_subgraphs(graph=graph)
        for subgraph in subgraphs:
            count = 0
            for neuron in subgraph:
                y = self.neuron_dynamics[neuron, :].copy() / max(self.neuron_dynamics[neuron, :])
                for j in range(len(y)):
                    y[j] = y[j] + 1.05 * count
                plt.plot(self.time, y, 'k', linewidth=1)
                plt.xticks([])
                plt.yticks([])
                count += 1
            plt.ylabel('ΔF/F')
            plt.xlabel('Time (s)')
            if title:
                plt.title(title)
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
    def get_graph(self, corr_mat=None, weighted=False):
        """
        Must pass a np.ndarray type object to corr_mat, or the Pearsons
        correlation matrix for the full dataset will be used.

        :param corr_mat:
        :param weighted:
        :return:
        """
        if not isinstance(corr_mat, np.ndarray):
            corr_mat = self.pearsons_correlation_matrix  # update to include other correlation metrics
        graph = nx.Graph()
        if weighted:
            for i in range(len(self.labels)):
                graph.add_node(str(self.labels[i]))
                for j in range(len(self.labels)):
                    if not i == j:
                        graph.add_edge(str(self.labels[i]), str(self.labels[j]), weight=corr_mat[i, j])
        else:
            for i in range(len(self.labels)):
                graph.add_node(str(self.labels[i]))
                for j in range(len(self.labels)):
                    if not i == j and corr_mat[i, j] > self.threshold:
                        graph.add_edge(str(self.labels[i]), str(self.labels[j]))
        return graph

    def get_random_graph(self, graph=None):
        """
        Generates a random graph. The nx.algorithms.smallworld.random_reference is adapted from the
        Maslov and Sneppen (2002) algorithm. It randomizes the existing graph.

        :return:
        """
        if graph is None:
            graph = self.get_graph()
        graph = nx.algorithms.smallworld.random_reference(graph)
        return graph

    def get_erdos_renyi_graph(self, graph=None):
        """
        Generates an Erdos-Renyi random graph using a graph edge coverage metric computed from the graph to be randomized.

        :param graph:
        :return:
        """
        if graph is None:
            num_nodes = self.num_neurons
            con_probability = self.get_graph_coverage()
        else:
            num_nodes = len(graph.nodes)
            con_probability = self.get_graph_coverage(graph=graph)
        return nx.erdos_renyi_graph(n=num_nodes, p=con_probability)

    # Todo: add functionality to improve plot
    def plot_graph(self, graph, show_labels=True, position=None):
        """

        :param show_labels:
        :param graph:
        :param position:
        :return:
        """
        if not position:
            position = nx.spring_layout(graph)
        nx.draw_networkx_nodes(graph, pos=position, node_color='b', node_size=100)
        nx.draw_networkx_edges(graph, pos=position, edge_color='b', )
        if show_labels:
            nx.draw_networkx_labels(graph, pos=position, font_size=6, font_color='w', font_family='sans-serif')
        plt.axis('off')
        plt.show()
        return

    # Todo: getting stuck on small world analysis when computing sigma -- infinite loop
    # Todo: this^ may be due to computing the average clustering coefficient or the average shortest path length -- test
    def get_smallworld_largest_subnetwork(self, graph=None):
        """

        :param graph:
        :return:
        """
        if graph is None:
            graph = self.get_largest_subgraph()
        else:
            graph = self.get_largest_subgraph(graph=graph)
        if len(graph.nodes()) >= 4:
            return nx.algorithms.smallworld.sigma(graph)
        else:
            raise RuntimeError(
                'Largest subgraph has less than four nodes. networkx.algorithms.smallworld.sigma cannot be computed.')

    # Todo: DO NOT USE FUNCTION, NOT UPDATED
    # def get_smallworld_all_subgraphs(self, corr_matrix, graph=None):
    #     """
    #
    #     :param corr_matrix:
    #     :param graph:
    #     :return:
    #     """
    #     if graph is None:
    #         graph = self.get_graph(corr_matrix)
    #     graph_max_subgraph_generator = sorted(nx.connected_components(graph), key=len, reverse=True)
    #     omega_list = []
    #     for i, val in enumerate(graph_max_subgraph_generator):
    #         graph_max_subgraph = graph.subgraph(val)
    #         if len(graph_max_subgraph.nodes) >= 4:
    #             omega = nx.algorithms.smallworld.omega(graph_max_subgraph)
    #             omega_list.append(omega)
    #     return omega_list

    # Todo: define hits_threshold based on tail of powerlaw distribution
    # Todo: determine best practices for setting threshold of powerlaw distribution to find hubs
    def get_hubs(self, graph=None):
        """
        Computes hub nodes in the graph and returns a list of nodes identified as hubs.

        :param graph:
        :return:
        """
        if graph is None:
            hubs, authorities = nx.hits(self.get_graph())
        else:
            hubs, authorities = nx.hits(graph)
        med_hubs = np.median(list(hubs.values()))
        std_hubs = np.std(list(hubs.values()))
        hubs_threshold = med_hubs + 2.5 * std_hubs
        hubs_list = []
        [hubs_list.append(x) for x in hubs.keys() if hubs[x] > hubs_threshold]
        return hubs_list, hubs

    def get_subgraphs(self, graph=None):
        """

        :param graph:
        :return:
        """
        if graph is None:
            connected_components = list(nx.connected_components(self.get_graph()))
        else:
            connected_components = list(nx.connected_components(graph))
        subgraphs = []
        [subgraphs.append(list(map(int, x))) for x in connected_components if len(x) > 1]
        return subgraphs

    def get_largest_subgraph(self, graph=None):
        """

        :param graph:
        :return:
        """
        if graph is None:
            graph = self.get_graph()
        largest_component = max(nx.connected_components(graph), key=len)
        return graph.subgraph(largest_component)

    # Todo: add functionality get_path_length
    def get_path_length(self):
        """
        Returns the characteristic path length.

        :return:
        """
        return

    def get_clustering_coefficient(self, graph=None) -> list:
        """
        Returns a list of clustering coefficient values for each node.
        
        :param graph:
        :return:
        """
        if graph is None:
            graph = self.get_graph_from_matrix()
        degree_view = nx.clustering(graph)
        clustering_coefficient = []
        [clustering_coefficient.append(degree_view[node]) for node in graph.nodes()]
        return clustering_coefficient

    # Todo: make this match the clustering coefficient
    def get_degree(self, graph=None):
        """
        Returns iterator object of (node, degree) pairs.

        :return:
        """
        if graph is None:
            return self.get_graph_from_matrix()
        else:
            return graph.degree

    # Todo: note that description is from
    def get_correlated_pair_ratio(self, graph=None):
        """
        Computes the number of connections each neuron has, divided by the nuber of cells in the field of view.
        This method is described in: https://www.nature.com/articles/s41467-020-17270-w#Sec8

        :param graph:
        :return:
        """
        if graph is None:
            graph = self.get_graph_from_matrix()
        degree_view = self.get_degree(graph)
        correlated_pair_ratio = []
        [correlated_pair_ratio.append(degree_view[node] / self.num_neurons) for node in graph.nodes()]
        return correlated_pair_ratio

    # Todo: adapt for directed - current total possible edges is for undirected
    def get_graph_coverage(self, graph=None):
        """
        Returns the percentage of edges present in the graph out of the total possible edges.

        :param graph:
        :return:
        """
        possible_edges = (self.num_neurons * (self.num_neurons - 1)) / 2
        if graph is None:
            graph = self.get_graph()
        return len(graph.edges) / possible_edges

    # Todo: check form of centrality
    def get_eigenvector_centrality(self, graph=None):
        """
        Compute the eigenvector centrality of all graph nodes, the
        measure of influence each node has on the graph.

        :param graph:
        :return:
        """
        if graph is None:
            graph = self.get_graph_from_matrix()
        centrality = nx.eigenvector_centrality(graph)
        return centrality

    def get_communities(self, graph=None) -> list:
        """
        Returns a list of communities, composed of a group of nodes.

        :param graph:
        :return: node_groups:
        """
        if graph is None:
            graph = self.get_graph_from_matrix()
        communities = nx.algorithms.community.centrality.girvan_newman(graph)
        node_groups = []
        for community in next(communities):
            node_groups.append(list(community))
        return node_groups

    # Todo: add additional arguments to expand functionality
    def draw_graph(self, graph=None, position=None, node_size=25, node_color='b', alpha=0.5):
        """
        Draws a simple graph.

        :param graph:
        :param position:
        :param node_size:
        :param node_color:
        :param alpha:
        :return:
        """
        if graph is None:
            graph = self.get_graph()
        if position is None:
            position = nx.spring_layout(graph)
        nx.draw(graph, pos=position, node_size=node_size, node_color=node_color, alpha=alpha)
