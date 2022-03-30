import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pynwb import NWBHDF5IO
import imageio
import os
from statsmodels.tsa.stattools import grangercausalitytests


class NeuronalNetworkGraph:
    """
    Ongoing development: 02/03/2022
    Published: XX/XX/XXXX
    Author: Veronica Porubsky [Github: https://github.com/vporubsky][ORCID: https://orcid.org/0000-0001-7216-3368]

    Class: NeuronalNetworkGraph(data_file, identifiers=None)
    =====================

    This class provides functionality to easily visualize time-series data of
    neuronal activity and to compute correlation metrics of neuronal networks,
    and generate graph objects which can be analyzed using graph theory.
    There are several graph theoretical metrics for further analysis of
    neuronal network connectivity patterns.
    ...

    Attributes
    ----------
    data_file : str
        A string pointing to the file to be used for data analysis.
    identifiers : list
        A list of identifiers for each row of calcium imaging data in
        the data_file passed to NeuronalNetworkGraph.

    Methods
    -------
    get_laplacian_matrix()
        ...
    Todo: **** Make additional classes within neuronal_network_graph so that analyses can be performed by
            subsampling many graphs or by doing simple divisions on a full dataset
            ex: class -- subsample_dataset
            ex usage: nng.subsample_dataset(subsample_indices = [(), (), ()]) -- return list of graphs
    Todo: Add additional graph theoretical analyses (path length, rich club, motifs...)
    Todo: Add functionality for parsing networks using context-active cell metadata.
    Todo: Add additional correlation metrics and allow user to pass them (currently must use Pearson)
    Todo: Add additional random network generators
    Todo: Add sliding window analysis ***
    Todo: Add timeshifting for analysis ***
    Todo: Consider adding converter to look at frequency domain or wavelets -- examine coherence
    Todo: Add function to generate stacked subnetwork/module timecourses
    Todo: Add documentation to all methods using docstrings
    Todo: Add all methods to Class docstring
    Todo: Determine the distribution of eigenvector centrality scores in connected modules/subnetworks
    Todo: Add justification for r=0.3 threshold: https://www.nature.com/articles/nature15389 https://www.nature.com/articles/nature12015
    Todo: ****** Implement shuffle distribution r value correction: https://www.nature.com/articles/s41467-020-17270-w#MOESM1
    Todo: Possible correlation method: https://www.frontiersin.org/articles/10.3389/fninf.2018.00007/full
    """

    def __init__(self, data_file, identifiers=None):
        """

        :param csv_file:
        :param identifiers:
        """
        if data_file.endswith('csv'):
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
        self.data_filename = str(data_file)
        self.time = self.data[0, :]
        self.neuron_dynamics = self.data[1:len(self.data), :]
        self.num_neurons = np.shape(self.neuron_dynamics)[0]
        if not identifiers:
            self.labels = np.linspace(0, np.shape(self.neuron_dynamics)[0] - 1, \
                                      np.shape(self.neuron_dynamics)[0]).astype(int)
        else:
            self.labels = np.array(identifiers)
        self.pearsons_correlation_matrix = np.corrcoef(self.neuron_dynamics)

        # Todo: move to DG class
        # Todo: A/B indices only for fear conditioning
        self.context_A_dynamics = self.neuron_dynamics[:, 1800:3600]  # Record second in Context A
        self.context_B_dynamics = self.neuron_dynamics[:, 0:1800]  # Record first in Context B
        self.con_A_pearsons_correlation_matrix = np.corrcoef(self.context_A_dynamics)
        self.con_B_pearsons_correlation_matrix = np.corrcoef(self.context_B_dynamics)

    # Todo: update description
    def get_laplacian_matrix(self, graph=None, threshold=0.3):
        """
        Returns the Laplacian matrix of the specified graph.

        :param graph:
        :param threshold:
        :return:
        """
        if not graph:
            graph = self.get_network_graph_from_matrix(threshold=threshold)
        return nx.laplacian_matrix(graph)

    # Todo: make this flexible enough to allow any matrix to be passed
    def get_network_graph_from_matrix(self, threshold=0.3, weight_matrix=None):
        """
        Automatically generate graph object from numpy adjacency matrix.

        :param weight_matrix:
        :param threshold:
        :param weighted:
        :return:
        """
        if weight_matrix is None:
            return nx.from_numpy_matrix(self.get_adjacency_matrix(threshold=threshold))
        return nx.from_numpy_matrix(self.get_weight_matrix(weight_matrix=weight_matrix))


    # Todo: may be superfluous
    def get_pearsons_correlation_matrix(self, data_matrix=None, time_points=None):
        """
        Returns the Pearson's correlation for all neuron pairs.

        :param data_matrix:
        :param time_points: tuple
        :return:
        """
        if not data_matrix:
            data_matrix = self.neuron_dynamics
        if time_points:
            data_matrix = data_matrix[:, time_points[0]:time_points[1]]
        return np.corrcoef(data_matrix, rowvar=True)

    # Todo: return list of graphs using specified time-subsampling
    # Todo: rename subsample_indices
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

    # Todo: refine adjacency matrix
    def get_adjacency_matrix(self, threshold=0.3):
        """

        :param threshold:
        :return:
        """
        adj_mat = (self.pearsons_correlation_matrix > threshold)
        np.fill_diagonal(adj_mat, 0)
        return adj_mat.astype(int)

    # Todo: update parameter description
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
        if not correlation_matrix:
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
        plt.ylabel('Fluorescence')
        plt.xlabel('Time (s)')
        if title is None:
            plt.title('Timecourse ' + str(neuron_timecourse_selection) + ' from ' + self.data_filename)
        else:
            plt.title(title)
        plt.show()

    # Todo: plot stacked timecourses based on input neuron indices from graph theory analyses
    # Todo: adjust y axis title for normalization
    # Todo add time ticks
    def plot_subnetworks_timecourses(self, graph=None, threshold=0.3, title=None):
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
            plt.ylabel('Fluorescence (norm)')
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
            plt.ylabel('Fluorescence')
            plt.xlabel('Time (s)')
            plt.title('Timecourse from ' + self.data_filename)
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

    def get_random_graph(self, threshold=0.3):
        """
        nx.algorithms.smallworld.random_reference is adapted from the Maslov and Sneppen (2002) algorithm.
        It uses an existing graph and randomizes it

        :param threshold:
        :return:
        """
        G = self.get_network_graph(threshold=threshold)
        G = nx.algorithms.smallworld.random_reference(G)
        return G

    # Todo: finish implementation of random graph
    def get_erdos_renyi_graph(self, graph=None, threshold=0.3):
        """
        Generates an Erdos-Renyi random graph using a network edge coverage
        metric computed from the graph to be randomized.

        :param graph:
        :param threshold:
        :return:
        """
        if not graph:
            num_nodes = self.num_neurons
            con_probability = self.get_network_coverage(threshold=threshold)
        else:
            num_nodes = len(graph.nodes)
            con_probability = self.get_network_coverage(graph=graph)
        erdos_renyi_rand_graph = nx.erdos_renyi_graph(n=num_nodes, p=con_probability)
        return erdos_renyi_rand_graph

    def plot_graph_network(self, graph, position):
        """

        :param graph:
        :param position:
        :return:
        """
        nx.draw_networkx_nodes(graph, pos=position, node_color='b', node_size=100)
        nx.draw_networkx_edges(graph, pos=position, edge_color='b', )
        nx.draw_networkx_labels(graph, pos=position, font_color='w', font_family='sans-serif')
        plt.axis('off')
        plt.show()
        return

        # Todo: write function

    # Todo: getting stuck on small world analysis when computing sigma -- infinite loop?
    # Todo: this^ may be due to computing the average clustering coefficient or the average shortest path length -- test
    def get_smallworld_largest_subnetwork(self, graph=None, threshold=0.3):
        """

        :param graph:
        :param threshold:
        :return:
        """
        if not graph:
            graph = self.get_largest_subnetwork_graph(threshold=threshold)
        else:
            graph = self.get_largest_subnetwork_graph(graph=graph, threshold=threshold)
        if len(graph.nodes()) >= 4:
            return nx.algorithms.smallworld.sigma(graph)
        else:
            raise RuntimeError(
                'Largest subnetwork has less than four nodes. networkx.algorithms.smallworld.sigma cannot be computed.')

    # Todo: DO NOT USE FUNCTION, NOT UPDATED
    def get_smallworld_all_subnetworks(self, corr_matrix, G=None, threshold=0.3):
        """

        :param corr_matrix:
        :param G:
        :param threshold:
        :return:
        """
        if not G:
            G = self.get_network_graph(corr_matrix, threshold)
        G_max_subgraph_generator = sorted(nx.connected_components(G), key=len, reverse=True)
        omega_list = []
        for i, val in enumerate(G_max_subgraph_generator):
            G_max_subgraph = G.subgraph(val)
            if len(G_max_subgraph.nodes) >= 4:
                omega = nx.algorithms.smallworld.omega(G_max_subgraph)
                omega_list.append(omega)
        return omega_list

    # Todo: consider using pagerank instead of HITS algorithm
    # Todo: rename graph argument
    # Todo: change hits to hubs
    # Todo: consider how best to allow the user to return hits_A or hits_B to do analysis
    # Todo: define hits_threshold based on tail of powerlaw distribution
    # Todo: determine best practices for setting threshold of powerlaw distribution to find hubs
    # Todo: ensure that it is ok to use median and standard deviation
    def get_hubs(self, graph=None, threshold=0.3):
        """

        :param graph:
        :param threshold:
        :return:
        """
        if not graph:
            hubs, authorities = nx.hits_numpy(self.get_network_graph(threshold=threshold))
        else:
            hubs, authorities = nx.hits_numpy(graph)
        med_hubs = np.median(list(hubs.values()))
        std_hubs = np.std(list(hubs.values()))
        hubs_threshold = med_hubs + 2.5 * std_hubs
        hubs_list = []
        [hubs_list.append(x) for x in hubs.keys() if hubs[x] > hubs_threshold]
        return hubs_list, hubs

    # Todo: subnetworks - all, change graph agument to something more meaningful
    # Todo: test get_subnetworks with no graph specified
    def get_subnetworks(self, graph=None, threshold=0.3):
        """

        :param graph:
        :param threshold:
        :return:
        """
        if not graph:
            connected_components = list(nx.connected_components(self.get_network_graph(threshold=threshold)))
        else:
            connected_components = list(nx.connected_components(graph))
        subnetworks = []
        [subnetworks.append(list(map(int, x))) for x in connected_components if len(x) > 1]
        return subnetworks

    # Todo: Return subnetwork graphs -- clean-up
    # Todo: decide whether to use "graph" or "G"
    def get_largest_subnetwork_graph(self, graph=None, threshold=0.3):
        """

        :param graph:
        :param threshold:
        :return:
        """
        if not graph:
            graph = self.get_network_graph(threshold=threshold)
        largest_component = max(nx.connected_components(graph), key=len)
        return graph.subgraph(largest_component)

    # Todo: review get_stability func
    def get_stability(self, num_folds, threshold):
        """

        :param num_folds:
        :param threshold:
        :return:
        """
        num_pts = int(np.floor(np.shape(self.neuron_dynamics)[1] / num_folds))
        r, c = np.shape(self.neuron_dynamics)
        stability_mat = np.zeros((r, r))
        for n in range(1, num_folds):
            corr_mat = np.corrcoef(self.neuron_dynamics[1:np.shape(self.neuron_dynamics)[0], \
                                   n * num_pts - num_pts:n * num_pts], \
                                   rowvar=True)
            r, c = np.shape(corr_mat)
            for row in range(r):
                for col in range(c):
                    if corr_mat[row, col] > threshold:
                        stability_mat[row, col] += 1
        percent_stability = stability_mat / num_folds
        neuron_1_index = []
        neuron_2_index = []
        for row in range(r):
            for col in range(c):
                if row != col:
                    if percent_stability[row, col] > 0.1:
                        neuron_1_index.append(row)
                        neuron_2_index.append(col)
        return percent_stability, (neuron_1_index, neuron_2_index)

    # Todo: write function
    def get_path_length(self):
        """
        Returns the characteristic path length.

        :return:
        """
        return

    # Todo: test function
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

    # Todo: decide if this is necessary for each context
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

    # Todo: note changed return form from: [correlated_pair_ratio.append(degree_view[node] / self.num_neurons) for node in graph.nodes()]
    # Todo: note that description is from https://www.nature.com/articles/s41467-020-17270-w#Sec8
    def get_correlated_pair_ratio(self, threshold=0.3, graph=None):
        """
        # pairs/ total # cells in FOV

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

    # Todo: get coverage -- figure out if coverage is the correct term
    # Todo: rename graph argument if necessary
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
        if not graph:
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
        if not graph:
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
        if not graph:
            graph = self.get_network_graph()
        nx.draw(graph, pos=nx.spring_layout(graph), node_size=node_size, node_color=node_color, alpha=alpha)


class DGNetworkGraph(NeuronalNetworkGraph):
    """
    Class for LC-DG experiments. Context A and Context B are specified for
    fear conditioning paradigm, where Context A is neutral and recorded from
    time 180 to 360 seconds and Context B is anxiogeneic and recorded from time
    0 to 180 seconds.
    """

    # Pass __init__ from parent class
    def __init__(self, data_file, identifiers=None):
        super().__init__(data_file, identifiers)
        self.context_A_dynamics = self.neuron_dynamics[:, 1800:3600]  # Record second in Context A
        self.context_B_dynamics = self.neuron_dynamics[:, 0:1800]  # Record first in Context B
        self.con_A_pearsons_correlation_matrix = np.corrcoef(self.context_A_dynamics)
        self.con_B_pearsons_correlation_matrix = np.corrcoef(self.context_B_dynamics)
    pass

    # Todo: Enable integration of cell-matched metadata
    def get_cell_matching_dict(self):
        """
        Returns a dictionary of neuron activity indices for day 1 and day 9
        for each mouse, for all neurons which remain in the field of view for
        both days.

        :return:
        """
        return

    def get_context_A_graph(self, threshold=0.3, weighted=False):
        """

        :param threshold:
        :return:
        """
        corr_mat = self.con_A_pearsons_correlation_matrix
        if weighted:
            return self.get_network_graph_from_matrix(weight_matrix=corr_mat)
        context_A_graph = nx.Graph()
        for i in range(len(self.labels)):
            context_A_graph.add_node(str(self.labels[i]))
            for j in range(len(self.labels)):
                if not i == j and corr_mat[i, j] > threshold:
                    context_A_graph.add_edge(str(self.labels[i]), str(self.labels[j]))
        return context_A_graph

    def get_context_B_graph(self, threshold=0.3, weighted=False):
        """

        :param threshold:
        :return:
        """
        corr_mat = self.con_B_pearsons_correlation_matrix
        if weighted:
            return self.get_network_graph_from_matrix(weight_matrix=corr_mat)
        context_B_graph = nx.Graph()
        for i in range(len(self.labels)):
            context_B_graph.add_node(str(self.labels[i]))
            for j in range(len(self.labels)):
                if not i == j and corr_mat[i, j] > threshold:
                    context_B_graph.add_edge(str(self.labels[i]), str(self.labels[j]))
        return context_B_graph

    def get_context_A_subnetworks(self, threshold=0.3):
        """

        :param threshold:
        :return:
        """
        connected_components_A = list(nx.connected_components(self.get_context_A_graph(threshold=threshold)))
        subnetwork_A = []
        [subnetwork_A.append(list(map(int, x))) for x in connected_components_A if len(x) > 1]
        return subnetwork_A

    def get_context_B_subnetworks(self, threshold=0.3):
        """

        :param threshold:
        :return:
        """
        connected_components_B = list(nx.connected_components(self.get_context_B_graph(threshold=threshold)))
        subnetwork_B = []
        [subnetwork_B.append(list(map(int, x))) for x in connected_components_B if len(x) > 1]
        return subnetwork_B

    def plot_subnetworks_A_timecourses(self, threshold=0.3):
        """

        :param threshold:
        :return:
        """
        subnetworks = self.get_context_A_subnetworks(threshold=threshold)
        for subnetwork in subnetworks:
            count = 0
            for neuron in subnetwork:
                y = self.neuron_dynamics[neuron, 1800:3600].copy() / max(self.neuron_dynamics[neuron, 1800:3600])
                for j in range(len(y)):
                    y[j] = y[j] + 1.05 * count
                plt.plot(self.time[1800:3600], y, 'k', linewidth=1)
                plt.xticks([])
                plt.yticks([])
                count += 1
            plt.ylabel('Fluorescence (norm)')
            plt.xlabel('Time (s)')
            plt.title(f'Subnetwork Timecourses - Context A - Threshold = {threshold}')
            plt.show()

    # Todo: adjust y axis title for normalization
    # Todo add time ticks
    def plot_subnetworks_B_timecourses(self, threshold=0.3):
        """

        :param threshold:
        :return:
        """
        subnetworks = self.get_context_B_subnetworks(threshold=threshold)
        for subnetwork in subnetworks:
            count = 0
            for neuron in subnetwork:
                y = self.neuron_dynamics[neuron, 0:1800].copy() / max(self.neuron_dynamics[neuron, 0:1800])
                for j in range(len(y)):
                    y[j] = y[j] + 1.05 * count
                plt.plot(self.time[0:1800], y, 'k', linewidth=1)
                plt.xticks([])
                plt.yticks([])
                count += 1
            plt.ylabel('Fluorescence (norm)')
            plt.xlabel('Time (s)')
            plt.title(f'Subnetwork Timecourses - Context B - Threshold = {threshold}')
            plt.show()

    def get_random_context_A_graph(self, threshold=0.3):
        """

        :param threshold:
        :return:
        """
        G = self.get_context_A_graph(threshold=threshold)
        random_A_graph = nx.algorithms.smallworld.random_reference(G)
        return random_A_graph

    def get_random_context_B_graph(self, threshold=0.3):
        """

        :param threshold:
        :return:
        """
        G = self.get_context_B_graph(threshold=threshold)
        random_B_graph = nx.algorithms.smallworld.random_reference(G)
        return random_B_graph

    def get_context_A_hubs(self, threshold=0.3):
        """

        :param threshold:
        :return:
        """
        hits_A, authorities_A = nx.hits_numpy(self.get_context_A_graph(threshold=threshold))
        med_hits = np.median(list(hits_A.values()))
        std_hits = np.std(list(hits_A.values()))
        hits_threshold = med_hits + 2.5 * std_hits
        hubs_A = []
        [hubs_A.append(x) for x in hits_A.keys() if hits_A[x] > hits_threshold]
        return hubs_A, hits_A

    def get_context_B_hubs(self, threshold=0.3):
        """

        :param threshold:
        :return:
        """
        hits_B, authorities_B = nx.hits_numpy(self.get_context_B_graph(threshold=threshold))
        med_hits = np.median(list(hits_B.values()))
        std_hits = np.std(list(hits_B.values()))
        hits_threshold = med_hits + 2.5 * std_hits
        hubs_B = []
        [hubs_B.append(x) for x in hits_B.keys() if hits_B[x] > hits_threshold]
        return hubs_B, hits_B

    def get_largest_context_A_subnetwork_graph(self, threshold=0.3):
        """

        :param threshold:
        :return:
        """
        G = self.get_context_A_graph(threshold=threshold)
        largest_component = max(nx.connected_components(G), key=len)
        return G.subgraph(largest_component)

    def get_largest_context_B_subnetwork_graph(self, threshold=0.3):
        """

        :param threshold:
        :return:
        """
        G = self.get_context_B_graph(threshold=threshold)
        largest_component = max(nx.connected_components(G), key=len)
        return G.subgraph(largest_component)

    def get_context_A_correlated_pair_ratio(self, threshold=0.3):
        """

        :param threshold:
        :return:
        """
        G = self.get_context_A_graph(threshold=threshold)
        degree_view = self.get_degree(G)
        correlated_pair_ratio = []
        [correlated_pair_ratio.append(degree_view[node] / self.num_neurons) for node in G.nodes()]
        return correlated_pair_ratio

    def get_context_B_correlated_pair_ratio(self, threshold=0.3):
        """

        :param threshold:
        :return:
        """
        G = self.get_context_B_graph(threshold=threshold)
        degree_view = self.get_degree(G)
        correlated_pair_ratio = []
        [correlated_pair_ratio.append(degree_view[node] / self.num_neurons) for node in G.nodes()]
        return correlated_pair_ratio

    def get_context_A_clustering_coefficient(self, threshold=0.3):
        """

        :param threshold:
        :return:
        """
        G = self.get_context_A_graph(threshold=threshold)
        degree_view = nx.clustering(G)
        clustering_coefficient = []
        [clustering_coefficient.append(degree_view[node]) for node in G.nodes()]
        return clustering_coefficient

    def get_context_B_clustering_coefficient(self, threshold=0.3):
        """

        :param threshold:
        :return:
        """
        G = self.get_context_B_graph(threshold=threshold)
        degree_view = nx.clustering(G)
        clustering_coefficient = []
        [clustering_coefficient.append(degree_view[node]) for node in G.nodes()]
        return clustering_coefficient

    def get_context_A_degree(self, threshold=0.3):
        """

        :param threshold:
        :return:
        """
        G = self.get_context_A_graph(threshold=threshold)
        return G.degree

    def get_context_B_degree(self, threshold=0.3):
        """

        :param threshold:
        :return:
        """
        G = self.get_context_B_graph(threshold=threshold)
        return G.degree


class BLANetworkGraph(NeuronalNetworkGraph):
    # Pass __init__ from parent class
    def __init__(self, csv_file, identifiers=None):
        super().__init__(csv_file, identifiers)
