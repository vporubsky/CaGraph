# CaGraph imports
import preprocess as prep
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pynwb import NWBHDF5IO
import pandas as pd
from IPython.display import display
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 150)
import os


# %% CaGraph class
class CaGraph:
    """
    Author: Veronica Porubsky
    Author Github: https://github.com/vporubsky
    Author ORCID: https://orcid.org/0000-0001-7216-3368

    Class: CaGraph(data_file, node_labels=None, node_metadata=None, dataset_id=None, threshold=None)
    =====================

    This class provides functionality to easily visualize time-series data of
    neuronal activity and to compute correlation metrics of neuronal networks,
    and generate graph objects which can be analyzed using graph theory.
    There are several graph theoretical metrics for further analysis of
    neuronal network connectivity patterns.

    Attributes
    ----------
    data : str or numpy.ndarray
        A string pointing to the file to be used for data analysis, or a numpy.ndarray containing data loaded into
        memory. The first (idx 0) row must contain timepoints, the subsequent rows each represent a single neuron timeseries
        of calcium fluorescence data sampled at the timepoints specified in the first row.
    node_labels: list
        A list of identifiers for each row of calcium imaging data (each neuron) in the data_file passed to CaGraph.
    node_metadata: dict
        Contains metadata which is associated with neurons in the network. Each key in the dictionary will be added as an
        attribute to the CaGraph object, and the associated value will be .
        Each value
    dataset_id: str
        A unique identifier can be added to the CaGraph object.
    threshold: float
        Sets a threshold to be used for thresholded graph.
    """

    def __init__(self, data, node_labels=None, node_metadata=None, dataset_id=None, threshold=None):
        """
        :param data: str
        :param node_labels: list
        :param node_metadata: dict
        :param dataset_id: str
        :param threshold: float
        """
        # Check that the input data is in the correct format and load dataset
        if isinstance(data, np.ndarray):
            self._data = data
            self._time = self._data[0, :]
            self._neuron_dynamics = self._data[1:len(self._data), :]
        elif isinstance(data, str):
            if data.endswith('csv'):
                self._data = np.genfromtxt(data, delimiter=",")
                self._time = self._data[0, :]
                self._neuron_dynamics = self._data[1:len(self._data), :]
            elif data.endswith('nwb'):
                with NWBHDF5IO(data, 'r') as io:
                    nwbfile_read = io.read()
                    nwb_acquisition_key = list(nwbfile_read.acquisition.keys())[0]
                    ca_from_nwb = nwbfile_read.acquisition[nwb_acquisition_key]
                    self._neuron_dynamics = ca_from_nwb.data[:]
                    self._time = ca_from_nwb.timestamps[:]
                    self._data = np.vstack((self._time, self._neuron_dynamics))
            else:
                raise TypeError('File path must have a .csv or .nwb file to load.')
        else:
            raise TypeError('Data must be passed as a str containing a .csv or .nwb file, or as numpy.ndarray.')

        # Add dataset identifier
        if dataset_id is not None:
            self._data_id = dataset_id

        # Compute time interval and number of neurons
        self._dt = self._time[1] - self._time[0]
        self._num_neurons = np.shape(self._neuron_dynamics)[0]

        # Generate node labels
        if node_labels is None:
            self._node_labels = np.linspace(0, np.shape(self._neuron_dynamics)[0] - 1,
                                            np.shape(self._neuron_dynamics)[0]).astype(int)
        else:
            self._node_labels = node_labels

        # Initialize correlation matrix, threshold, and graph
        self._pearsons_correlation_matrix = self.get_pearsons_correlation_matrix()
        if threshold is not None:
            self._threshold = threshold
        else:
            self._threshold = self.__generate_threshold()
        self._graph = self.get_graph()

        # Store initial settings to reset attributes after modification
        self.__init_threshold = self._threshold
        self.__init_pearsons_correlation_matrix = self._pearsons_correlation_matrix
        self.__init_graph = self._graph

        # Initialize subclass objects
        self.graph_theory = self.GraphTheory(neuron_dynamics=self._neuron_dynamics, time=self._time,
                                             num_neurons=self._num_neurons,
                                             pearsons_correlation_matrix=self._pearsons_correlation_matrix,
                                             graph=self._graph, labels=self._node_labels)

        self.plotting = self.Plotting(neuron_dynamics=self._neuron_dynamics, time=self._time,
                                      num_neurons=self._num_neurons,
                                      pearsons_correlation_matrix=self._pearsons_correlation_matrix, graph=self._graph)

        # Initialize base graph theory analyses
        self._degree = self.graph_theory.get_degree(return_type='dict')
        self._clustering_coefficient = self.graph_theory.get_clustering_coefficient(return_type='dict')
        self._correlated_pair_ratio = self.graph_theory.get_correlated_pair_ratio(return_type='dict')
        self._communities = self.graph_theory.get_communities(return_type='dict')
        self._hubs = self.graph_theory.get_hubs(return_type='dict')
        self._hits = self.graph_theory.get_hits_values(return_type='dict')
        self._eigenvector_centrality = self.graph_theory.get_eigenvector_centrality(return_type='dict')

        # Build private attribute dictionary
        self.__attribute_dictionary = {'hubs': self.hubs, 'degree': self.degree,
                                       'clustering coefficient': self.clustering_coefficient,
                                       'communities': self.communities,
                                       'eigenvector centrality': self.eigenvector_centrality,
                                       'correlated pair ratio': self.correlated_pair_ratio, 'HITS': self.hits}

        # Add node metadata
        if node_metadata is not None:
            for key in node_metadata.keys():
                if type(node_metadata[key]) is list or np.ndarray:
                    if len(node_metadata[key]) != len(self._node_labels):
                        raise ValueError('Each key-value pair in the node_metadata dictionary must have a value to be '
                                         'associated with every node.')
                    node_metadata_dict = {}
                    for i, value in enumerate(node_metadata[key]):
                        node_metadata_dict[self._node_labels[i]] = value
                    self.__attribute_dictionary[key] = node_metadata_dict
                    setattr(self, key, node_metadata_dict)
                elif type(node_metadata[key]) is dict:
                    self.__attribute_dictionary[key] = node_metadata
                    setattr(self, key, node_metadata[key])
                else:
                    raise AttributeError('Each key-value pair in the node_metadata dictionary must have a value'
                                         'supplied as type list, one-dimensional numpy.ndarray, or as a dictionary'
                                         'where each key is a node and each value is the metadata value for that node.')

    # Private utility methods
    @property
    def data(self):
        return self._data

    @property
    def time(self):
        return self._time

    @property
    def neuron_dynamics(self):
        return self._neuron_dynamics

    @property
    def dt(self):
        return self._dt

    @property
    def num_neurons(self):
        return self._num_neurons

    @property
    def data_id(self):
        return self._data_id

    @property
    def node_labels(self):
        return self._node_labels

    @property
    def pearsons_correlation_matrix(self):
        return self._pearsons_correlation_matrix

    @property
    def graph(self):
        return self._graph

    @property
    def degree(self):
        return self._degree

    @property
    def clustering_coefficient(self):
        return self._clustering_coefficient

    @property
    def correlated_pair_ratio(self):
        return self._correlated_pair_ratio

    @property
    def communities(self):
        return self._communities

    @property
    def hubs(self):
        return self._hubs

    @property
    def hits(self):
        return self._hits

    @property
    def eigenvector_centrality(self):
        return self._eigenvector_centrality

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value
        self._pearsons_correlation_matrix = self.get_pearsons_correlation_matrix()
        self._graph = self.get_graph()
        self._degree = self.graph_theory.get_degree(return_type='dict')
        self._clustering_coefficient = self.graph_theory.get_clustering_coefficient(return_type='dict')
        self._correlated_pair_ratio = self.graph_theory.get_correlated_pair_ratio(return_type='dict')
        self._communities = self.graph_theory.get_communities(return_type='dict')
        self._hubs = self.graph_theory.get_hubs(return_type='dict')
        self._hits = self.graph_theory.get_hits_values(return_type='dict')
        self._eigenvector_centrality = self.graph_theory.get_eigenvector_centrality(return_type='dict')

    def __generate_threshold(self) -> float:
        """
        Generates a threshold for the provided dataset as described in the preprocess module.

        :return: float
        """
        return prep.generate_average_threshold(data=self._neuron_dynamics, shuffle_iterations=10)

    def __parse_by_node(self, node_data, node_list) -> list:
        """
        Method to parse report analyses using only a subset of nodes.

        :param node_data: list
        :param node_list: list
        :return: list
        """
        return [node_data[i] for i in node_list if i < len(node_data)]

    # Public utility methods
    def reset(self):
        """
        Resets the CaGraph object graph attribute to the original state at the time the object was created.
        """
        self._pearsons_correlation_matrix = self.__init_pearsons_correlation_matrix
        self._threshold = self.__init_threshold
        self._graph = self.__init_graph

        # Re-initialize base graph theory analyses
        self._degree = self.graph_theory.get_degree()
        self._clustering_coefficient = self.graph_theory.get_clustering_coefficient()
        self._correlated_pair_ratio = self.graph_theory.get_correlated_pair_ratio()
        self._communities = self.graph_theory.get_communities()
        self._hubs = self.graph_theory.get_hubs()

    def show_dataset(self):
        """
        Uses pandas to show the dataset in a Jupyter notebook.

        """
        data_df = pd.DataFrame(self.data)
        build_index = ['time']
        for i in range(self.data.shape[0] - 1):
            build_index.append('neuron ' + str(i))
        data_df.index = build_index
        display(data_df.style)

    # Statistics and linear algebra methods
    def get_pearsons_correlation_matrix(self, data_matrix=None) -> np.ndarray:
        """
        Returns the Pearson's correlation for all neuron pairs.

        A loaded numpy.ndarray dataset can be passed to the method for analysis, otherwise
        the dataset passed to the CaGraph object constructor will be used.

        :param data_matrix: numpy.ndarray
        :return:
        """
        if data_matrix is None:
            data_matrix = self._neuron_dynamics
        return np.nan_to_num(np.corrcoef(data_matrix, rowvar=True))

    # Todo: determine if you need this to be adjusted so you can set multiple thresholds (< 0.1, > 0.5)
    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Returns the adjacency matrix of a graph where edges exist when greater than the provided threshold.

        Uses the Pearson's correlation matrix.

        :return: numpy.ndarray
        """
        adj_mat = (self._pearsons_correlation_matrix > self._threshold).astype(int)
        np.fill_diagonal(adj_mat, 0)
        return adj_mat

    def get_laplacian_matrix(self, graph=None) -> np.ndarray:
        """
        Returns the Laplacian matrix of the specified graph.

        :param graph: networkx.Graph object
        :return:
        """
        if graph is None:
            graph = self.get_graph()
        return nx.laplacian_matrix(graph).toarray()

    def get_weight_matrix(self) -> np.ndarray:
        """
        Returns a weighted connectivity matrix with zero along the diagonal. No threshold is applied.

        :return: numpy.ndarray
        """
        weight_matrix = self._pearsons_correlation_matrix
        np.fill_diagonal(weight_matrix, 0)
        return weight_matrix

    # Graph construction methods
    def get_graph(self, weighted=False) -> nx.Graph:
        """
        Automatically generate graph object from numpy adjacency matrix.

        :param weighted: bool
        :return: networkx.Graph object
        """
        if not weighted:
            return nx.from_numpy_array(self.get_adjacency_matrix())
        return nx.from_numpy_array(self.get_weight_matrix())

    def get_random_graph(self, graph=None) -> nx.Graph:
        """
        Generates a random graph. The nx.algorithms.smallworld.random_reference is adapted from the
        Maslov and Sneppen (2002) algorithm. It randomizes the existing graph.

        :type graph: networkx.Graph object
        :return: networkx.Graph object
        """
        if graph is None:
            graph = self.get_graph()
        graph = nx.algorithms.smallworld.random_reference(graph)
        return graph

    def get_erdos_renyi_graph(self, graph=None) -> nx.Graph:
        """
        Generates an Erdos-Renyi random graph using a graph edge density metric computed from the graph to be randomized.

        :param graph:
        :return: networkx.Graph object
        """
        if graph is None:
            num_nodes = self._num_neurons
            con_probability = self.graph_theory.get_graph_density()
        else:
            num_nodes = len(graph.nodes)
            con_probability = self.graph_theory.get_graph_density(graph=graph)
        return nx.erdos_renyi_graph(n=num_nodes, p=con_probability)

    # Todo: add additional arguments to expand functionality
    # Todo: add show and save functionality
    def draw_graph(self, graph=None, position=None, node_size=25, node_color='b', alpha=0.5):
        """
        Draws a simple graph.

        :param graph: networkx.Graph object
        :param position: dict
        :param node_size: int
        :param node_color: str
        :param alpha: float
        :return:
        """
        if graph is None:
            graph = self._graph
        if position is None:
            position = nx.spring_layout(graph)
        nx.draw(graph, pos=position, node_size=node_size, node_color=node_color, alpha=alpha)

    # Todo: add option to generate a directory with analysis files
    def get_report(self, parsing_nodes=None, parse_by_attribute=None, parsing_operation=None,
                   parsing_value=None, save_report=False, save_path=None, save_filename=None, save_filetype=None):
        """
        :param save_filetype:
        :param save_filename:
        :param save_path:
        :param save_report:
        :param parsing_nodes:
        :param parse_by_attribute: str
        :param parsing_operation: str
        :param parsing_value: float
        :return: dict
        """
        # Set up parsing
        if parse_by_attribute is not None:
            # identify nodes that meet the parsing criteria
            if parsing_operation == '>':
                parsing_nodes = [key for key, value in self.__attribute_dictionary[parse_by_attribute].items() if
                                 value > parsing_value]
            elif parsing_operation == '<':
                parsing_nodes = [key for key, value in self.__attribute_dictionary[parse_by_attribute].items() if
                                 value < parsing_value]
            elif parsing_operation == '<=':
                parsing_nodes = [key for key, value in self.__attribute_dictionary[parse_by_attribute].items() if
                                 value <= parsing_value]
            elif parsing_operation == '>=':
                parsing_nodes = [key for key, value in self.__attribute_dictionary[parse_by_attribute].items() if
                                 value >= parsing_value]
            elif parsing_operation == '==':
                parsing_nodes = [key for key, value in self.__attribute_dictionary[parse_by_attribute].items() if
                                 value == parsing_value]
            elif parsing_operation == '!=':
                parsing_nodes = [key for key, value in self.__attribute_dictionary[parse_by_attribute].items() if
                                 value != parsing_value]

        # Individual node analyses
        if parsing_nodes is None:
            parsing_nodes = self._node_labels
        report_dict = {}
        for key in self.__attribute_dictionary.keys():
            report_dict[key] = self.__parse_by_node(node_data=self.__attribute_dictionary[key], node_list=parsing_nodes)

        # Construct report dataframe
        report_df = pd.DataFrame.from_dict(report_dict, orient='columns')
        report_df.index = parsing_nodes

        # Save report
        if save_report:
            if save_filename is None:
                save_filename = 'report'
            if save_path is None:
                save_path = os.getcwd() + '/'
            if save_filetype is None or save_filetype == 'csv':
                report_df.to_csv(save_path + save_filename + '.csv', index=True)
            elif save_filetype == 'HDF5':
                report_df.to_hdf(save_path + save_filename + '.h5', key=save_filename, mode='w')
            elif save_filetype == 'xlsx':
                report_df.to_excel(save_path + save_filename + 'xlsx', index=True)
        return report_df

    class GraphTheory:
        def __init__(self, neuron_dynamics, time, pearsons_correlation_matrix, graph, num_neurons, labels):
            self._time = time
            self._neuron_dynamics = neuron_dynamics
            self._num_neurons = num_neurons
            self._pearsons_correlation_matrix = pearsons_correlation_matrix
            self._graph = graph
            self._node_labels = labels

        # Private utility methods
        @property
        def time(self):
            return self._time

        @property
        def neuron_dynamics(self):
            return self._neuron_dynamics

        @property
        def num_neurons(self):
            return self._num_neurons

        @property
        def graph(self):
            return self._graph

        @property
        def node_labels(self):
            return self._node_labels

        @property
        def pearsons_correlation_matrix(self):
            return self._pearsons_correlation_matrix

        # Graph theory analysis
        def get_hubs(self, graph=None, return_type='list'):
            """
            Computes hub nodes in the graph using the HITS algorithm. Hubs are identified by finding
            those nodes which have a hub value greater than the median of the hubs values plus 2.5 time the standard deviation.

            :param graph: networkx.Graph object
            :param return_type: str
            """
            if graph is None:
                hubs, authorities = nx.hits(self._graph, max_iter=500)
            else:
                hubs, authorities = nx.hits(graph, max_iter=500)
            med_hubs = np.median(list(hubs.values()))
            std_hubs = np.std(list(hubs.values()))
            hubs_threshold = med_hubs + 2.5 * std_hubs
            hubs_list = []
            [hubs_list.append(x) for x in hubs.keys() if hubs[x] > hubs_threshold]
            hub_dict = {i: 1 if i in list(set(hubs_list) & set(self._node_labels)) else 0 for i in self._node_labels}
            if return_type == 'dict':
                return hub_dict
            if return_type == 'list':
                return list(hub_dict.values())

        def get_hits_values(self, graph=None, return_type='list'):
            """
            Computes hub nodes in the graph and returns a list of nodes identified as hubs.
            HITS and authorities values match due to bidirectional edges.

            :param graph: networkx.Graph object
            :param return_type: str
            """
            if graph is None:
                graph = self._graph
            hits_dict, authorities_dict = nx.hits(graph, max_iter=500)
            if return_type == 'dict':
                return hits_dict
            if return_type == 'list':
                return list(hits_dict.values())

        def get_connected_components(self, graph=None) -> list:
            """
            Returns connected components with more than one node.

            :param graph: networkx.Graph object
            :return: list
            """
            if graph is None:
                connected_components_with_orphan = list(nx.connected_components(self._graph))
            else:
                connected_components_with_orphan = list(nx.connected_components(graph))
            connected_components = []
            [connected_components.append(list(map(int, x))) for x in connected_components_with_orphan if len(x) > 1]
            return connected_components

        def get_largest_connected_component(self, graph=None) -> nx.Graph:
            """
            Returns a subgraph containing the largest connected component.

            :param graph: networkx.Graph object
            :return: networkx.Graph object
            """
            if graph is None:
                graph = self._graph
            largest_component = max(nx.connected_components(graph), key=len)
            return graph.subgraph(largest_component)

        # Todo: add functionality get_path_length
        def get_path_length(self):
            """
            Returns the characteristic path length.

            :return:
            """
            return

        def get_clustering_coefficient(self, graph=None, return_type='list'):
            """
            Returns a list of clustering coefficient values for each node.

            :param graph: networkx.Graph object
            :param return_type: str
            """
            if graph is None:
                graph = self._graph
            degree_view = nx.clustering(graph)
            clustering_coefficient = []
            [clustering_coefficient.append(degree_view[node]) for node in graph.nodes()]
            clustering_dict = dict(zip(self._node_labels, clustering_coefficient))
            if return_type == 'dict':
                return clustering_dict
            if return_type == 'list':
                return list(clustering_dict.values())

        def get_degree(self, graph=None, return_type='list'):
            """
            Returns iterator object of (node, degree) pairs.

            :param graph: networkx.Graph object
            :param return_type: str
            """
            if graph is None:
                graph = self._graph
            degree_dict = dict(graph.degree)
            if return_type == 'dict':
                return degree_dict
            if return_type == 'list':
                return list(degree_dict.values())

        def get_correlated_pair_ratio(self, graph=None, return_type='list'):
            """
            Computes the number of connections each neuron has, divided by the nuber of cells in the field of view.
            This method is described in Jimenez et al. 2020: https://www.nature.com/articles/s41467-020-17270-w#Sec8

            :param graph: networkx.Graph object
            :param return_type: str
            """
            if graph is None:
                graph = self._graph
            degree_view = self.get_degree(graph)
            correlated_pair_ratio = []
            [correlated_pair_ratio.append(degree_view[node] / self._num_neurons) for node in graph.nodes()]
            correlated_pair_ratio_dict = dict(zip(self._node_labels, correlated_pair_ratio))
            if return_type == 'dict':
                return correlated_pair_ratio_dict
            if return_type == 'list':
                return list(correlated_pair_ratio_dict.values())

        # Todo: adapt for directed - current total possible edges is for undirected
        def get_graph_density(self, graph=None):
            """
            Returns the ratio of edges present in the graph out of the total possible edges.
            The equation used to calculate this ratio is only relevant to undirected graphs.

            :param graph: networkx.Graph object
            :return: float
            """
            possible_edges = (self._num_neurons * (self._num_neurons - 1)) / 2
            if graph is None:
                graph = self._graph
            return len(graph.edges) / possible_edges

        def get_eigenvector_centrality(self, graph=None, return_type='list'):
            """
            Compute the eigenvector centrality of all graph nodes, the
            measure of influence each node has on the graph.

            :param graph: networkx.Graph object
            :param return_type: str
            """
            if graph is None:
                graph = self._graph
            eigenvector_centrality = nx.eigenvector_centrality(graph)
            if return_type == 'dict':
                return eigenvector_centrality
            if return_type == 'list':
                return list(eigenvector_centrality.values())

        def get_communities(self, graph=None, return_type='list'):
            """
            Returns a list of communities, composed of a group of nodes.

            :param return_type: list
            :param graph: networkx.Graph object
            :return: node_groups: list
            """
            if graph is None:
                graph = self._graph
            communities = list(nx.algorithms.community.greedy_modularity_communities(graph))
            sorted(communities)
            community_id = {}
            for i in range(len(communities)):
                for j in list(communities[i]):
                    community_id[j] = i
            if return_type == 'dict':
                return community_id
            if return_type == 'list':
                return list(community_id.values())

        # Todo: getting stuck on small world analysis when computing sigma -- infinite loop may be due to computing the average clustering coefficient or the average shortest path length -- test
        def get_smallworld_largest_subnetwork(self, graph=None) -> float:
            """

            :param graph: networkx.Graph object
            :return: float
            """
            if graph is None:
                graph = self.get_largest_connected_component()
            else:
                graph = self.get_largest_connected_component(graph=graph)
            if len(graph.nodes()) >= 4:
                return nx.algorithms.smallworld.sigma(graph)
            else:
                raise RuntimeError(
                    'Largest subgraph has less than four nodes. networkx.algorithms.smallworld.sigma cannot be computed.')

        # Todo: high priority add function
        def compare_graphs(self):
            """

            :return:
            """

    class Plotting:
        def __init__(self, neuron_dynamics, time, pearsons_correlation_matrix, graph, num_neurons):
            self._num_neurons = num_neurons
            self._time = time
            self._neuron_dynamics = neuron_dynamics
            self._pearsons_correlation_matrix = pearsons_correlation_matrix
            self._graph = graph

        # Private utility methods
        @property
        def time(self):
            return self._time

        @property
        def neuron_dynamics(self):
            return self._neuron_dynamics

        @property
        def num_neurons(self):
            return self._num_neurons

        @property
        def graph(self):
            return self._graph

        @property
        def pearsons_correlation_matrix(self):
            return self._pearsons_correlation_matrix

        def plot_correlation_heatmap(self, correlation_matrix=None, title=None, y_label=None, x_label=None,
                                     show_plot=True,
                                     save_plot=False, save_path=None, dpi=300, save_format='png'):
            """
            Plots a heatmap of the correlation matrix.

            :param save_format:
            :param correlation_matrix:
            :param title:
            :param y_label:
            :param x_label:
            :param show_plot:
            :param save_plot:
            :param save_path:
            :param dpi:
            :return:
            """
            if correlation_matrix is None:
                correlation_matrix = self._pearsons_correlation_matrix()
            sns.heatmap(correlation_matrix, vmin=0, vmax=1)
            if title is not None:
                plt.title(title)
            if y_label is not None:
                plt.ylabel(y_label)
            if x_label is not None:
                plt.xlabel(x_label)
            if show_plot:
                plt.show()
            if save_plot:
                if save_path is None:
                    save_path = os.getcwd() + f'fig'
                plt.savefig(fname=save_path, dpi=dpi, format=save_format)

        def get_single_neuron_timecourse(self, neuron_trace_number) -> np.ndarray:
            """
            Return time vector stacked on the recorded calcium fluorescence for the neuron of interest.

            :param neuron_trace_number: int
            :return: numpy.ndarray
            """
            neuron_timecourse_selection = neuron_trace_number
            return np.vstack((self._time, self._neuron_dynamics[neuron_timecourse_selection, :]))

        def plot_single_neuron_timecourse(self, neuron_trace_number, title=None, y_label=None, x_label=None,
                                          show_plot=True,
                                          save_plot=False, save_path=None, dpi=300, save_format='png'):
            """

            :param save_format:
            :param neuron_trace_number: int
            :param title:
            :param y_label:
            :param x_label:
            :param show_plot:
            :param save_plot:
            :param save_path:
            :param dpi:
            :return:
            """
            neuron_timecourse_selection = neuron_trace_number
            count = 1
            x_tick_array = []
            for i in range(len(self._time)):
                if count % (len(self._time) / 20) == 0:
                    x_tick_array.append(self._time[i])
                count += 1
            plt.figure(num=1, figsize=(10, 2))
            plt.plot(self._time, self._neuron_dynamics[neuron_timecourse_selection, :],
                     linewidth=1)  # add option : 'xkcd:olive',
            plt.xticks(x_tick_array)
            plt.xlim(0, self._time[-1])
            if title is not None:
                plt.title(title)
            if y_label is not None:
                plt.ylabel(y_label)
            else:
                plt.ylabel('ΔF/F')
            if x_label is not None:
                plt.xlabel(x_label)
            else:
                plt.xlabel('Time')
            if show_plot:
                plt.show()
            if save_plot:
                if save_path is None:
                    save_path = os.getcwd() + f'fig'
                plt.savefig(fname=save_path, dpi=dpi, format=save_format)

        def plot_multi_neuron_timecourse(self, neuron_trace_labels, palette=None, title=None, y_label=None,
                                         x_label=None,
                                         show_plot=True, save_plot=False, save_path=None, dpi=300, save_format='png'):
            """
            Plots multiple individual calcium fluorescence traces, stacked vertically.


            :param save_format:
            :param neuron_trace_labels: list
            :param palette: list
            :param title:
            :param y_label:
            :param x_label:
            :param show_plot:
            :param save_plot:
            :param save_path:
            :param dpi:
            :return:
            """
            count = 0
            if palette is None:
                palette = sns.color_palette('husl', len(neuron_trace_labels))
            for idx, neuron in enumerate(neuron_trace_labels):
                y = self._neuron_dynamics[neuron, :].copy() / max(self._neuron_dynamics[neuron, :])
                y = [x + 1.05 * count for x in y]
                plt.plot(self._time, y, c=palette[idx], linewidth=1)
                plt.xticks([])
                plt.yticks([])
                count += 1
            if title is not None:
                plt.title(title)
            if y_label is not None:
                plt.ylabel(y_label)
            else:
                plt.ylabel('ΔF/F')
            if x_label is not None:
                plt.xlabel(x_label)
            else:
                plt.xlabel('Time')
            if show_plot:
                plt.show()
            if save_plot:
                if save_path is None:
                    save_path = os.getcwd() + f'fig'
                plt.savefig(fname=save_path, dpi=dpi, format=save_format)

        def plot_all_neurons_timecourse(self, title=None, y_label=None, x_label=None, show_plot=True, save_plot=False,
                                        save_path=None, dpi=300, save_format='png'):
            """
            :param save_format:
            :param title:
            :param y_label:
            :param x_label:
            :param show_plot:
            :param save_plot:
            :param save_path:
            :param dpi:
            """
            plt.figure(num=2, figsize=(10, 2))
            count = 1
            x_tick_array = []
            for i in range(len(self._time)):
                if count % (len(self._time) / 20) == 0:
                    x_tick_array.append(self._time[i])
                count += 1
            for i in range(len(self._neuron_dynamics) - 1):
                plt.plot(self._time, self._neuron_dynamics[i, :], linewidth=0.5)
                plt.xticks(x_tick_array)
                plt.xlim(0, self._time[-1])
            if title is not None:
                plt.title(title)
            if y_label is not None:
                plt.ylabel(y_label)
            else:
                plt.ylabel('ΔF/F')
            if x_label is not None:
                plt.xlabel(x_label)
            else:
                plt.xlabel('Time')
            if show_plot:
                plt.show()
            if save_plot:
                if save_path is None:
                    save_path = os.getcwd() + f'fig'
                plt.savefig(fname=save_path, dpi=dpi, format=save_format)


# %% Time samples
class CaGraphTimeSamples:
    """
    Class for running time-sample analyses on a single dataset.
    """
    def __init__(self, data, time_samples=None, condition_labels=None, node_labels=None, node_metadata=None,
                 dataset_id=None,
                 threshold=None):
        """
        :param data: str
        :param time_samples:
        :param condition_labels:
        :param node_labels: list
        :param node_metadata: dict
        :param dataset_id: str
        :param threshold: float
        """
        # Check that the input data is in the correct format and load dataset
        if isinstance(data, np.ndarray):
            self._data = data
        elif isinstance(data, str):
            if data.endswith('csv'):
                self._data = np.genfromtxt(data, delimiter=",")
            elif data.endswith('nwb'):
                with NWBHDF5IO(data, 'r') as io:
                    nwbfile_read = io.read()
                    nwb_acquisition_key = list(nwbfile_read.acquisition.keys())[0]
                    ca_from_nwb = nwbfile_read.acquisition[nwb_acquisition_key]
                    self._data = np.vstack((ca_from_nwb.timestamps[:], ca_from_nwb.data[:]))
            else:
                raise TypeError('File path must have a .csv or .nwb file to load.')
        else:
            raise TypeError('Data must be passed as a str containing a .csv or .nwb file, or as numpy.ndarray.')

        # Add dataset identifier
        if dataset_id is not None:
            self._data_id = dataset_id

        self._condition_identifiers = condition_labels

        # Compute time interval and number of neurons
        self._dt = self.data[0, 1] - self.data[0, 0]
        self._num_neurons = np.shape(self.data)[0]
        if threshold is not None:
            self._threshold = threshold
        else:
            self._threshold = self.__generate_threshold()

        # Generate node labels
        if node_labels is None:
            self._node_labels = np.linspace(0, np.shape(self.data)[0] - 2,
                                            np.shape(self.data)[0] - 1).astype(int)
        else:
            self._node_labels = node_labels

        # Add a series of private attributes which are CaGraph objects
        for i, sample in enumerate(time_samples):
            setattr(self, f'__{condition_labels[i]}_cagraph',
                    CaGraph(data=self._data[:, sample[0]:sample[1]], node_labels=self._node_labels,
                            node_metadata=node_metadata, threshold=self._threshold))

    # Private utility methods
    @property
    def data(self):
        return self._data

    @property
    def dt(self):
        return self._dt

    @property
    def num_neurons(self):
        return self._num_neurons

    @property
    def data_id(self):
        return self._data_id

    @property
    def node_labels(self):
        return self._node_labels

    @property
    def threshold(self):
        return self._threshold

    @property
    def condition_identifiers(self):
        return self._condition_identifiers

    def __generate_threshold(self) -> float:
        """
        Generates a threshold for the provided dataset as described in the preprocess module.
        This threshold generation will use the full dataset.

        :return: float
        """
        return prep.generate_average_threshold(data=self.data[1:,:], shuffle_iterations=10)

    # Public utility methods
    def get_cagraph(self, condition_label):
        """

        :param condition_label:
        :return:
        """
        return getattr(self, f'__{condition_label}_cagraph')

    def get_full_report(self, save_report=False, save_path=None, save_filename=None, save_filetype=None):
        """
        Generates an organized report of all data in the batched sample. It will report on the
        base analyses included in the CaGraph object get_report() method, and output a single
        pandas DataFrame or file which includes these analyses for all datasets in a tabular structure.

        :param save_report:
        :param save_path:
        :param save_filename:
        :param save_filetype:
        :return:
        """
        store_reports = {}
        for key in self._condition_identifiers:
            cagraph_obj = self.get_cagraph(key)
            store_reports[key] = cagraph_obj.get_report()

        # For each column in the individual reports, append to the full report
        full_report_df = pd.DataFrame()
        for col in store_reports[key].columns:
            for key in store_reports.keys():
                df = store_reports[key]
                df = df.rename(columns={col: f'{key}_{col}'})
                full_report_df = pd.concat([full_report_df, df[f'{key}_{col}']], axis=1)

        # Save the report
        if save_report:
            if save_filename is None:
                save_filename = 'report'
            if save_path is None:
                save_path = os.getcwd() + '/'
            if save_filetype is None or save_filetype == 'csv':
                full_report_df.to_csv(save_path + save_filename + '.csv', index=True)
            elif save_filetype == 'HDF5':
                full_report_df.to_hdf(save_path + save_filename + '.h5', key=save_filename, mode='w')
            elif save_filetype == 'xlsx':
                full_report_df.to_excel(save_path + save_filename + 'xlsx', index=True)
        return full_report_df


# %% Batched analyses
class CaGraphBatch:
    """
    Class for running batched analyses.

    Only directories can be passed to CaGraphBatch. Node metadata cannot be added to CaGraph objects in the batched
    analysis. Future versions will include the node_metadata attribute.
    """

    def __init__(self, data_path, group_id=None, threshold=None, threshold_averaged=False):
        """
        Path to data must be specified with data_path. A group identifier can optionally be specified with group_id.
        The threshold can be set in three ways - 1. manually set by the user at the time of object creation,
        2. if not set manually, all

        :param group_id: str
        :param threshold: float
        """
        if not os.path.exists(os.path.dirname(data_path)):
            raise ValueError('Path provided for data_path parameter does not exist.')
        data_list = os.listdir(data_path)

        # Set threshold
        if threshold is not None:
            self._threshold = threshold
        elif threshold_averaged:
            threshold_keys = []
            for dataset in data_list:
                if dataset.endswith(".csv"):
                    threshold_keys.append(dataset[:-4])
            self._threshold = self.__generate_averaged_threshold(data_path=data_path, dataset_keys=threshold_keys)
        else:
            self._threshold = None
        if group_id is not None:
            self._group_id = group_id

        # Construct CaGraph objects for each dataset
        self._dataset_identifiers = []
        for dataset in data_list:
            if dataset.endswith(".csv"):
                data = np.genfromtxt(data_path + dataset, delimiter=",")
                try:
                    setattr(self, f'__{dataset[:-4]}_cagraph', CaGraph(data=data, threshold=self._threshold))
                    self._dataset_identifiers.append(dataset[:-4])
                except Exception as e:
                    print(f"Exception occurred for dataset {dataset[:-4]}: " + repr(e))

    # Private utility methods
    @property
    def group_id(self):
        return self._group_id

    @property
    def threshold(self):
        return self._threshold

    @property
    def dataset_identifiers(self):
        return self._dataset_identifiers

    def __generate_averaged_threshold(self, data_path, dataset_keys):
        """
        Computes an averaged threshold by computing the mean of the recommended thresholds for each individual dataset.

        :param data_path:
        :param dataset_keys:
        :return:
        """
        store_thresholds = []
        for dataset in dataset_keys:
            data = np.genfromtxt(f'{data_path}{dataset}.csv', delimiter=",")
            store_thresholds.append(prep.generate_average_threshold(data=data[1:,:], shuffle_iterations=10))
        return np.mean(store_thresholds)

    # Public utility methods
    def get_cagraph(self, condition_label) -> CaGraph:
        """
        Return a CaGraph object for the specified dataset condition_label

        :param condition_label: str
        :return: CaGraph
        """
        return getattr(self, f'__{condition_label}_cagraph')

    def get_full_report(self, save_report=False, save_path=None, save_filename=None, save_filetype=None):
        """
        Generates an organized report of all data in the batched sample. It will report on the
        base analyses included in the CaGraph object get_report() method, and output a single
        pandas DataFrame or file which includes these analyses for all datasets in a tabular structure.

        :param save_report:
        :param save_path:
        :param save_filename:
        :param save_filetype:
        :return:
        """
        store_reports = {}
        for key in self.dataset_identifiers:
            cagraph_obj = self.get_cagraph(key)
            store_reports[key] = cagraph_obj.get_report()

        # For each column in the individual reports, append to the full report
        full_report_df = pd.DataFrame()
        for col in store_reports[key].columns:
            for key in store_reports.keys():
                df = store_reports[key]
                df = df.rename(columns={col: f'{key}_{col}'})
                full_report_df = pd.concat([full_report_df, df[f'{key}_{col}']], axis=1)

        # Save the report
        if save_report:
            if save_filename is None:
                save_filename = 'report'
            if save_path is None:
                save_path = os.getcwd() + '/'
            if save_filetype is None or save_filetype == 'csv':
                full_report_df.to_csv(save_path + save_filename + '.csv', index=True)
            elif save_filetype == 'HDF5':
                full_report_df.to_hdf(save_path + save_filename + '.h5', key=save_filename, mode='w')
            elif save_filetype == 'xlsx':
                full_report_df.to_excel(save_path + save_filename + 'xlsx', index=True)
        return full_report_df

    def save_individual_dataset_reports(self, save_path=None, save_filetype=None):
        """
        Saves individual reports for each of the specified datasets.
        Individual filenames will be generated using the filename name of the dataset from which the analysis is derived.

        This will result in the same analysis that can be done by creating a CaGraph object using a single dataset.

        :param save_path: str
        :param save_filetype: str ('csv', 'HDF5', 'xlsx')
        :return:
        """
        # Iterate through all datasets and save report for each
        for key in self.dataset_identifiers:
            cagraph_obj = self.get_cagraph(key)
            cagraph_obj.get_report(save_report=True, save_path=save_path, save_filename=key + '_report',
                                   save_filetype=save_filetype)

# %%  Matched analyses




# %% Remaining updates
# Todo: CaGraphBatch -> allow user to specify labels and also pass loaded numpy arrays
# Todo: CaGraphBatch -> add option to add cell metadata
# Todo: CaGraphBatch -> ensure that datasets which are thrown out at initial CaGraph object generation are not included in future versions
# Todo: CaGraphBatch -> make a constructor function that can pass loaded data, numpy arrays
# Todo: All classes -> add check that all save_paths exist, otherwise create them
# Todo: All classes -> check docstrings again
# Todo: dd whole-graph analysis like report_dict['density']  = self.graph_theory.get_graph_density()
# Todo: CaGraphBatch -> high priority write second report method that averages results and stores the averages
# Todo: CaGraphTimeSamples -> add checks that length of time samples and condition labels are equal -- user guardrails
# Todo: CaGraphTimeSamples -> Create a systematic return report/ dictionary