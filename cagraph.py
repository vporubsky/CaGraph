# CaGraph imports
from scipy.spatial.distance import correlation

import preprocess
import visualization
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import Ridge
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pynwb import NWBHDF5IO
import pandas as pd
import os
import pickle


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

    def __init__(self, data, correlation_matrix=None, correlation_method='pearson', node_labels=None,
                 node_metadata=None, dataset_id=None, threshold=None, **correlation_kwargs):
        """
        Initialize a CaGraph object from calcium imaging data or an existing correlation matrix.

        Parameters:
        ----------
        data : str or np.ndarray
            Path to a .csv or .nwb file, or a numpy array. The first row must contain timepoints;
            subsequent rows represent calcium fluorescence traces from individual neurons.

        correlation_matrix : np.ndarray, optional
            Precomputed correlation or similarity matrix to use instead of computing from raw data.

        correlation_method : str, optional (default='pearson')
            Method used to compute the correlation matrix if none is provided. Options include:
                - 'pearson': Linear correlation
                - 'spearman': Rank-based correlation (monotonic relationships)
                - 'crosscorr': Max cross-correlation across time lags
                - 'partial': Direct correlation between neuron pairs controlling for all others
                - 'mutual_info': Nonlinear statistical dependence (if implemented)
                - 'granger': Directional influence (if implemented)

        node_labels : list, optional
            List of identifiers for each neuron in the dataset (excluding the time row).

        node_metadata : dict, optional
            Dictionary mapping metadata keys to lists of per-neuron values. Will be attached to the CaGraph object.

        dataset_id : str, optional
            Unique string identifier for the dataset (useful for labeling or saving).

        threshold : float, optional
            Correlation threshold (0–1) to apply when generating a thresholded graph.

        correlation_kwargs : keyword arguments
            Additional keyword arguments specific to the selected `correlation_method`. For example:
                - crosscorr_max_lag (int): maximum lag for 'crosscorr' (default: 10)
                - partial_alpha (float): regularization strength for 'partial' (default: 1.0)

        Returns:
        -------
        CaGraph object
            Initialized and optionally thresholded graph based on calcium imaging data.
        """
        # Check that the input data is in the correct format and load dataset
        self.__input_validator(data=data)

        # Add dataset identifier
        if dataset_id is not None:
            self._dataset_id = dataset_id
        else:
            self._dataset_id = None

        # Compute time interval and number of neurons
        self._dt = self._data[0, 1] - self._data[0, 0]
        self._num_neurons = np.shape(self._neuron_dynamics)[0]

        # Generate node labels
        if node_labels is None:
            self._node_labels = np.linspace(0, np.shape(self._neuron_dynamics)[0] - 1,
                                            np.shape(self._neuron_dynamics)[0]).astype(int)
        else:
            self._node_labels = node_labels

        # Initialize coordinates attribute
        self.coordinates = None

        # Initialize correlation matrix, threshold, and graph
        if correlation_matrix is not None:
            self._correlation_matrix = correlation_matrix
        else:
            self._correlation_matrix = self.get_correlation_matrix(data_matrix=self._neuron_dynamics,
                                                                   method=correlation_method,
                                                                   **correlation_kwargs)

        if threshold is not None:
            self._threshold = threshold
        else:
            self._threshold = self.__generate_threshold(correlation_method=correlation_method, **correlation_kwargs)
        self._graph = self.get_graph(threshold=self._threshold)

        # Store initial settings to reset attributes after modification
        self.__init_threshold = self._threshold
        self.__init_correlation_matrix = self._correlation_matrix
        self.__init_graph = self._graph

        # Initialize subclass objects
        self.analysis = self.Analysis(neuron_dynamics=self._neuron_dynamics, time=self._time,
                                      num_neurons=self._num_neurons,
                                      correlation_matrix=self._correlation_matrix,
                                      graph=self._graph, labels=self._node_labels)

        self.plotting = self.Plotting(neuron_dynamics=self._neuron_dynamics, time=self._time,
                                      num_neurons=self._num_neurons,
                                      correlation_matrix=self._correlation_matrix, graph=self._graph)

        # Todo: expand visualization capabilities
        self.visualization = self.Visualization(cagraph_obj=self)

        # Initialize base graph theory analyses
        # Todo: add try except block for adding each analysis? so more can be included that may generate some errors?
        self._degree = self.analysis.get_degree(return_type='dict')
        self._clustering_coefficient = self.analysis.get_clustering_coefficient(return_type='dict')
        self._correlated_pair_ratio = self.analysis.get_correlated_pair_ratio(return_type='dict')
        self._communities = self.analysis.get_communities(return_type='dict')
        self._hubs = self.analysis.get_hubs(return_type='dict')
        self._betweenness_centrality = self.analysis.get_betweenness_centrality(return_type='dict')
        # Todo: check eigenvector centrality convergence error
        # self._eigenvector_centrality = self.graph_theory.get_eigenvector_centrality(return_type='dict')

        # Build private attribute dictionary
        self.__attribute_dictionary = {'hubs': self.hubs, 'degree': self.degree,
                                       'clustering coefficient': self.clustering_coefficient,
                                       'communities': self.communities,
                                       # Todo: decide if best to remove eigenvector centrality
                                       # 'eigenvector centrality': self.eigenvector_centrality,
                                       'correlated pair ratio': self.correlated_pair_ratio,
                                       'betweenness centrality': self.betweenness_centrality}

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
    def dataset_id(self):
        return self._dataset_id

    @property
    def node_labels(self):
        return self._node_labels

    @property
    def correlation_matrix(self):
        return self._correlation_matrix

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
    def betweenness_centrality(self):
        return self._betweenness_centrality

    # Todo: check eigenvector centrality convergence error
    # @property
    # def eigenvector_centrality(self):
    #     return self._eigenvector_centrality

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        if not (0 <= value <= 1):
            raise ValueError("Threshold must be between 0 and 1.")
        self._threshold = value
        self._graph = self.get_graph(threshold=self._threshold)
        self._degree = self.analysis.get_degree(return_type='dict')
        self._clustering_coefficient = self.analysis.get_clustering_coefficient(return_type='dict')
        self._correlated_pair_ratio = self.analysis.get_correlated_pair_ratio(return_type='dict')
        self._communities = self.analysis.get_communities(return_type='dict')
        self._hubs = self.analysis.get_hubs(return_type='dict')
        self._betweenness_centrality = self.analysis.get_betweenness_centrality(return_type='dict')
        # Todo: check eigenvector centrality convergence error
        # self._eigenvector_centrality = self.graph_theory.get_eigenvector_centrality(return_type='dict')

    def __input_validator(self, data):
        """
        Validate and load the input data for analysis.

        :param data: str or numpy.ndarray
            A string pointing to the file to be used for data analysis, or a numpy.ndarray containing data loaded into
            memory. The first (idx 0) row must contain timepoints, the subsequent rows each represent a single neuron
            timeseries of calcium fluorescence data sampled at the timepoints specified in the first row.
        """
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

    def __generate_threshold(self, correlation_method='pearson', **correlation_kwargs) -> float:
        """
        Generates a threshold for the provided dataset as described in the preprocess module.

        :return: float
            The computed threshold.
        """
        return preprocess.generate_average_threshold(data=self._neuron_dynamics, shuffle_iterations=10, correlation_method=correlation_method, **correlation_kwargs)

    def __parse_by_node(self, node_data, node_list) -> list:
        """
        Parse and return data from a list using only a subset of nodes.

        This method takes a list of node data and a list of node indices to extract data for specific nodes.
        It returns a new list containing the data for the specified nodes. The method ensures that the indices
        in 'node_list' are within the valid range of the 'node_data' list.

        :param node_data: list
            A list of data to be filtered based on node indices.
        :param node_list: list
            A list of node indices to specify which data to extract.

        :return: list
            A new list containing data for the specified nodes.
        """
        return [node_data[i] for i in node_list if i < len(node_data)]

    # Public utility methods
    def reset(self):
        """
        Reset the CaGraph object to its initial state.

        This method restores the CaGraph object to its original state as it was when created, including the
        initial graph, correlation matrix, and threshold. It also re-initializes the base graph theory analyses.

        Usage:
        ```
        cagraph = CaGraph(data, node_labels, node_metadata, dataset_id, threshold)
        # Make changes to the object
        cagraph.reset()  # Restore the object to its initial state
        ```

        """
        self._correlation_matrix = self.__init_correlation_matrix
        self._threshold = self.__init_threshold
        self._graph = self.__init_graph

        # Re-initialize base graph theory analyses
        self._degree = self.analysis.get_degree(return_type='dict')
        self._clustering_coefficient = self.analysis.get_clustering_coefficient(return_type='dict')
        self._correlated_pair_ratio = self.analysis.get_correlated_pair_ratio(return_type='dict')
        self._communities = self.analysis.get_communities(return_type='dict')
        self._hubs = self.analysis.get_hubs(return_type='dict')

    def save(self, file_path=None):
        """
        Save the CaGraph object to a file.

        This method allows you to save the current state of the CaGraph object to a file. You can specify the 'file_path'
        parameter to choose the location and name of the saved file. If 'file_path' is not provided, the object is saved with
        a default file name based on the dataset ID or as 'obj.cagraph' if no dataset ID is set.

        :param file_path: str, optional
            The path to the file where the CaGraph object will be saved. If not provided, a default file name will be used.
        """
        if file_path is None:
            if self.dataset_id is not None:
                file_path = self.dataset_id + '.cagraph'
            else:
                file_path = 'obj.cagraph'
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path):
        """
        Load a CaGraph object from a saved file.

        This static method allows you to load a previously saved CaGraph object from a file. You should provide the 'file_path'
        parameter, which is the path to the file containing the serialized CaGraph object.

        :param file_path: str
            The path to the file from which the CaGraph object will be loaded.

        :return: CaGraph
            The loaded CaGraph object.
        """
        with open(file_path, 'rb') as file:
            cagraph_obj = pickle.load(file)
        return cagraph_obj

    def load_coordinates(self, path):
        """
        Load neuron spatial coordinates from a CSV or Excel file.
        Returns a dict: {node_id: (x, y)}
        """
        if path.endswith(".csv"):
            data = np.genfromtxt(path, delimiter=',', encoding='utf-8-sig')
            if data.shape[0] != 2:
                raise ValueError("CSV must contain exactly 2 rows: x and y coordinates.")
            x_coords, y_coords = data
        elif path.endswith(".xlsx"):
            df = pd.read_excel(path, header=None)
            if df.shape[0] != 2:
                raise ValueError("Excel must contain exactly 2 rows: x and y coordinates.")
            x_coords = df.iloc[0].to_numpy()
            y_coords = df.iloc[1].to_numpy()
        else:
            raise ValueError("Unsupported file type. Use .csv or .xlsx")

        if len(x_coords) != len(y_coords):
            raise ValueError("Mismatch between x and y coordinate lengths.")

        self.coordinates = {i: (x, y) for i, (x, y) in enumerate(zip(x_coords, y_coords))}

    def sensitivity_analysis(self, data, correlation_method='pearson', interval=None, threshold=None, show_plot=True, save_plot=False, save_path=None,
                             dpi=300, save_format='png', **correlation_kwargs):
        """
            Perform sensitivity analysis on the CaGraph object by generating a series of graphs with varying thresholds and
            measuring their similarity to the original graph using graph edit distance.

            This method calculates a series of thresholds based on the provided 'interval' or a default set of threshold values.
            For each threshold, it generates a new graph and computes the graph edit distance between the new graph and the
            original graph. The results are visualized as a plot of thresholds versus graph edit distances.

            :param data: str or numpy.ndarray
                The data for the sensitivity analysis. This can be a string pointing to a data file or a numpy.ndarray
                containing the data.
            :param interval: list, optional
                A list of threshold values to be analyzed. If not provided, a default interval is used.
            :param threshold: float, optional
                The initial threshold for the analysis. If not provided, it is calculated from the 'data'.
            :param show_plot: bool, optional
                Determines whether to display the sensitivity analysis plot. Default is True.
            :param save_plot: bool, optional
                Determines whether to save the sensitivity analysis plot. Default is False.
            :param save_path: str, optional
                The path to save the plot if 'save_plot' is set to True. If not provided, the plot is saved in the current working
                directory with a default filename.
            :param dpi: int, optional
                The DPI (dots per inch) for the saved plot. Default is 300.
            :param save_format: str, optional
                The format for the saved plot file (e.g., 'png', 'jpg'). Default is 'png'.

            :return: list
                A list of similarity values (graph edit distances) between the original graph and the series of generated graphs.
            """
        if threshold is None:
            threshold = preprocess.generate_threshold(data=data,correlation_method=correlation_method, **correlation_kwargs)

        starting_graph = self.get_graph(threshold=threshold)
        if interval is not None:
            pass
        else:
            interval = [-0.3, -0.2, -0.1, -0.05, -0.025, -0.01, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        thresholds = []
        similarity = []
        for value in interval:
            if 0 < threshold + value <= 1:
                thresholds.append(threshold + value)
                similarity.append(nx.graph_edit_distance(starting_graph, self.get_graph(threshold=threshold + value)))

        plt.plot(thresholds, similarity, '.-')
        plt.xlabel('threshold')
        plt.ylabel('Graph edit distance')
        if save_plot:
            if save_path is None:
                save_path = os.getcwd() + f'fig'
            elif not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(fname=save_path, dpi=dpi, format=save_format)
        if show_plot:
            plt.show()
        return similarity

    # Statistics and linear algebra methods
    def get_correlation_matrix(self, method="pearson", data_matrix=None, crosscorr_max_lag=10,
                               partial_alpha=1.0) -> np.ndarray:
        """
        Calculate a correlation or similarity matrix between neurons using the specified method.

        Supported Methods:
        - 'pearson': Measures linear correlation between neuron activity traces.
        - 'spearman': Measures rank-based (monotonic) correlation, robust to outliers.
        - 'crosscorr': Computes maximum cross-correlation across time lags (requires 'crosscorr_max_lag').
        - 'partial': Estimates direct correlation between neurons by controlling for all others using Ridge regression (requires 'partial_alpha').

        Parameters:
        - method (str): One of the supported methods listed above.
        - data_matrix (np.ndarray): Optional override for internal data (shape: neurons × timepoints).
        - crosscorr_max_lag (int, optional): Maximum lag to consider in cross-correlation (default: 10).
        - partial_alpha (float, optional): Regularization strength for partial correlation via Ridge regression (default: 1.0).

        Returns:
        - np.ndarray: Correlation or similarity matrix of shape (neurons × neurons).
        """
        data_matrix = data_matrix if data_matrix is not None else self._neuron_dynamics
        n = data_matrix.shape[0]
        corr_matrix = np.ones((n, n))
        if method == "pearson":
            return np.nan_to_num(np.corrcoef(data_matrix, rowvar=True))
        elif method == "spearman":
            return np.nan_to_num(stats.spearmanr(data_matrix.T).correlation)
        elif method == "crosscorr":
            for i in range(n):
                for j in range(i + 1, n):
                    x, y = data_matrix[i], data_matrix[j]
                    corr = max([np.corrcoef(np.roll(x, lag), y)[0, 1] for lag in
                                range(-crosscorr_max_lag, crosscorr_max_lag + 1)])
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
            return corr_matrix
        elif method == "partial":
            for i in range(n):
                for j in range(i + 1, n):
                    idx = [k for k in range(n) if k != i and k != j]
                    Z = data_matrix[idx].T
                    model_i = Ridge(alpha=partial_alpha).fit(Z, data_matrix[i])
                    model_j = Ridge(alpha=partial_alpha).fit(Z, data_matrix[j])
                    res_i = data_matrix[i] - model_i.predict(Z)
                    res_j = data_matrix[j] - model_j.predict(Z)
                    corr = np.corrcoef(res_i, res_j)[0, 1]
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
            return corr_matrix
        else:
            raise ValueError(f"Unsupported method: {method}")

    def get_adjacency_matrix(self, threshold=None) -> np.ndarray:
        """
        Calculate and return the adjacency matrix of the graph based on the provided threshold.

        The adjacency matrix represents connections between nodes (neurons) in the graph. A value of 1 indicates
        an edge between nodes when the correlation is greater than the specified threshold, while a value
        of 0 indicates no edge.

        :param threshold: float, optional
            The threshold for determining edge existence. If not provided, the threshold specified during object
            initialization is used.

        :return: numpy.ndarray
            The adjacency matrix as a numpy array.
        """
        if threshold is None:
            adj_mat = (self._correlation_matrix > self._threshold).astype(int)
        else:
            adj_mat = (self._correlation_matrix > threshold).astype(int)
        np.fill_diagonal(adj_mat, 0)
        return adj_mat

    def get_laplacian_matrix(self, graph=None) -> np.ndarray:
        """
        Calculate and return the Laplacian matrix of the specified graph.

        The Laplacian matrix is a mathematical representation of a graph that encodes important structural information
        about the graph. If no 'graph' parameter is provided, the Laplacian matrix of the CaGraph object's graph is calculated.

        :param graph: networkx.Graph, optional
            The graph for which to compute the Laplacian matrix. If not provided, the CaGraph object's graph is used.

        :return: numpy.ndarray
            The Laplacian matrix as a numpy array.
        """
        if graph is None:
            graph = self.get_graph(threshold=self._threshold)
        return nx.laplacian_matrix(graph).toarray()

    def get_weight_matrix(self) -> np.ndarray:
        """
        Calculate and return a weighted connectivity matrix with zeros along the diagonal.

        This method returns a connectivity matrix representing weighted connections between nodes (neurons) in the graph.
        The diagonal elements are set to zero to represent self-connections, and no threshold is applied to the matrix.

        :return: numpy.ndarray
            The weighted connectivity matrix as a numpy array.
        """
        weight_matrix = self._correlation_matrix
        np.fill_diagonal(weight_matrix, 0)
        return weight_matrix

    # Graph construction methods
    def get_graph(self, threshold=None, weighted=False) -> nx.Graph:
        """
        Automatically generate a graph object based on the specified adjacency or weighted matrix.

        This method generates a networkx.Graph object from either the adjacency matrix (which could be reduced using a threshold) or the
        weighted connectivity matrix, depending on the 'weighted' parameter.

        :param threshold: float, optional
            The threshold for determining edge existence when generating the graph from the adjacency matrix. Ignored if
            'weighted' is True.
        :param weighted: bool, optional
            If True, a weighted graph is generated from the weighted connectivity matrix. If False, a binary (thresholded)
            graph is generated based on the adjacency matrix.

        :return: networkx.Graph
            The generated graph object.
        """
        if not weighted:
            return nx.from_numpy_array(self.get_adjacency_matrix(threshold=threshold))
        return nx.from_numpy_array(self.get_weight_matrix())

    def get_random_graph(self, graph=None) -> nx.Graph:
        """
        Generate a random graph based on the existing graph structure using the Maslov and Sneppen (2002) algorithm.

        This method generates a new random graph based on the provided 'graph' or the CaGraph object's graph. The randomization
        process is adapted from the Maslov and Sneppen algorithm using the 'nx.algorithms.smallworld.random_reference' function.

        :param graph: networkx.Graph, optional
            The graph to use as a basis for generating the random graph. If not provided, the CaGraph object's graph is used.

        :return: networkx.Graph
            The generated random graph.
        """
        if graph is None:
            graph = self.get_graph()
        graph = nx.algorithms.smallworld.random_reference(graph)
        return graph

    def get_erdos_renyi_graph(self, graph=None) -> nx.Graph:
        """
        Generate an Erdos-Renyi random graph based on the graph's edge density.

        This method generates an Erdos-Renyi random graph with 'n' nodes and edge probability 'p'. The values 'n' and 'p' are
        computed from the provided 'graph' or the CaGraph object's graph. If 'graph' is not provided, the number of nodes and
        edge density are calculated from the CaGraph object's graph structure.

        :param graph: networkx.Graph, optional
            The graph to use as a basis for calculating 'n' and 'p'. If not provided, the CaGraph object's graph is used.

        :return: networkx.Graph
            The generated Erdos-Renyi random graph.
        """
        if graph is None:
            num_nodes = self._num_neurons
            edge_probability = self.analysis.get_density()
        else:
            num_nodes = len(graph.nodes)
            edge_probability = self.analysis.get_density(graph=graph)
        return nx.erdos_renyi_graph(n=num_nodes, p=edge_probability)

# Todo: fix draw labels - what if user specifies label?
    def draw_graph(
            self,
            graph=None,
            position=None,
            node_size=25,
            node_color='b',
            alpha=0.5,
            show_labels=False,
            show=True,
            save_path=None,
            **kwargs
    ):
        """
        Visualize and optionally save a networkx.Graph with a given layout or spatial coordinates.

        Parameters:
        - graph: networkx.Graph (default: self._graph)
        - position: dict {node_id: (x, y)} or None
        - node_size: int
        - node_color: str or list
        - alpha: float
        - show: bool (whether to display the plot)
        - save_path: str or None (file path to save figure; supports .png, .pdf, etc.)
        - kwargs: passed to nx.draw_networkx_nodes
        """
        if graph is None:
            graph = self._graph
        if position is None:
            position = self.coordinates or nx.spring_layout(graph)

        plt.figure(figsize=(8, 6))

        nx.draw_networkx_nodes(
            graph,
            position,
            node_size=node_size,
            node_color=node_color,
            edgecolors='black',
            linewidths=0.8,
            alpha=alpha,
            **kwargs
        )
        nx.draw_networkx_edges(
            graph,
            position,
            edge_color='gray',
            width=1.2,
            alpha=0.5
        )
        if show_labels:
            nx.draw_networkx_labels(
                graph,
                position,
                font_size=6,
                font_color='black'
            )

        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    # Todo: add option to only select a subset of the graph analyses -- analysis_selections
    def get_report(self, parsing_nodes=None, parse_by_attribute=None, parsing_operation=None, parsing_value=None,
                   analysis_selections=None, save_report=False, save_path=None, save_filename=None, save_filetype=None):
        """
            Generate a report for the CaGraph object based on specified criteria and save it as a file if desired.

            This method constructs a report by selecting and analyzing data from the CaGraph object's node attributes.
            You can specify criteria for selecting nodes, parsing attributes, and filtering data. The resulting report
            is returned as a pandas DataFrame. If 'save_report' is set to True, the report can also be saved as a file.

            :param analysis_selections: list, optional
                This list contains str which denote the graph theory analysis to be included, ex. ['hubs', 'clustering coefficient']
            :param save_filetype: str, optional
                The file format to use for saving the report ('csv', 'HDF5', 'xlsx'). Default is 'csv'.
            :param save_filename: str, optional
                The name of the saved report file. Default is 'report'.
            :param save_path: str, optional
                The directory path for saving the report. Default is the current working directory.
            :param save_report: bool, optional
                Determines whether to save the report as a file. Default is False.
            :param parsing_nodes: list, optional
                A list of node labels to include in the report. Default is None, including all nodes.
            :param parse_by_attribute: str, optional
                The attribute name to parse and include in the report. Default is None, which includes all attributes.
            :param parsing_operation: str, optional
                The operation to apply when parsing the attribute data ('>', '<', '<=', '>=', '==', '!='). Default is None.
            :param parsing_value: float, optional
                The value used in the parsing operation. Default is None.

            :return: pd.DataFrame
                The report as a pandas DataFrame.
            """
        # Todo: check that filetype is valid
        valid_filetypes = {'csv', 'HDF5', 'xlsx'}  # Add all valid file types here

        if save_report and save_filetype not in valid_filetypes:
            raise ValueError(f"Invalid save file type: {save_filetype}")

        # Set up parsing
        if parse_by_attribute is not None:
            if parsing_operation is None or parsing_value is None:
                raise ValueError(
                    "Arguments 'parsing_operation' and 'parsing_value' must be specified if 'parse_by_attribute' is also specified.")

            # identify nodes that meet the parsing criteria
            else:
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

        if analysis_selections is not None:
            # Todo: check that this works and add to get_full_report/ other get_report methods -- checking that analysis selections are valid
            if len(analysis_selections) == 0:
                raise ValueError("Analysis selections cannot be empty.")
            invalid_selections = set(analysis_selections) - self.__attribute_dictionary.keys()
            if invalid_selections:
                raise ValueError(f"Invalid analysis selections: {', '.join(invalid_selections)}")
            selections_str = '|'.join(analysis_selections)
            report_df = report_df.filter(regex=selections_str)

            # Save report
        if save_report:
            if save_filename is None:
                save_filename = 'report'
            if save_path is None:
                save_path = os.getcwd() + '/'
            elif not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            if save_filetype is None or save_filetype == 'csv':
                report_df.to_csv(save_path + save_filename + '.csv', index=True)
            elif save_filetype == 'HDF5':
                report_df.to_hdf(save_path + save_filename + '.h5', key=save_filename, mode='w')
            elif save_filetype == 'xlsx':
                report_df.to_excel(save_path + save_filename + '.xlsx', index=True, engine='openpyxl')
        return report_df

    class Analysis:
        def __init__(self, neuron_dynamics, time, correlation_matrix, graph, num_neurons, labels):
            self._time = time
            self._neuron_dynamics = neuron_dynamics
            self._num_neurons = num_neurons
            self._correlation_matrix = correlation_matrix
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
        def correlation_matrix(self):
            return self._correlation_matrix

        # Graph theory analysis - global network structure
        def get_density(self, graph=None):
            """
            Calculate and return the density of the network.

            The density of a network is defined as the ratio of the number of edges present in the graph to the total number
            of possible edges in the graph. A higher density indicates a denser network with more connections.

            :param graph: networkx.Graph, optional
                The graph for which to calculate density. If not provided, the CaGraph object's graph is used.

            :return: float
                The density of the network.
            """
            if graph is None:
                graph = self._graph
            return nx.density(graph)

        def get_shortest_path_length(self, graph=None, source=None, target=None):
            """
            Calculate and return the shortest path length between two nodes in the network.

            This method computes the shortest path length between a source and target node in the network graph. The shortest path
            length represents the minimum number of edges that must be traversed to reach the target node from the source node.

            :param graph: networkx.Graph, optional
                The graph in which to calculate the shortest path. If not provided, the CaGraph object's graph is used.
            :param source: node
                The source node from which to calculate the shortest path.
            :param target: node
                The target node to which the shortest path is calculated.

            :return: int
                The shortest path length between the source and target nodes.
            """
            if graph is None:
                graph = self._graph
            return nx.shortest_path_length(graph, source=source, target=target)

        # Graph theory analysis - local network structure
        def get_hubs(self, graph=None, return_type='list'):
            """
            Calculate and return hub nodes in the network based on normalized betweenness centrality scores.

            This method identifies hub nodes by calculating the normalized betweenness centrality scores for all nodes in the
            network graph. Hub nodes are nodes with betweenness centrality scores exceeding an outlier threshold, which is
            determined using the inter-quartile range (IQR) of the betweenness centrality score distribution.

            :param graph: networkx.Graph, optional
                The graph for which to calculate hub nodes. If not provided, the CaGraph object's graph is used.
            :param return_type: str, optional
                The format in which to return the hub nodes. 'list' returns a list of hub nodes, 'dict' returns a dictionary with
                hub nodes marked as 1 and non-hub nodes as 0.

            :return: list or dict
                The hub nodes in the network, either as a list or a dictionary, based on the specified 'return_type'.
            """
            if graph is None:
                graph = self._graph

            # Calculate betweenness centrality
            betweenness_centrality = nx.betweenness_centrality(graph, normalized=True, endpoints=False)
            betweenness_centrality_scores = list(betweenness_centrality.values())
            betweenness_centrality_scores.sort()

            # Compute the outlier threshold using the interquartile range
            Q1 = np.percentile(betweenness_centrality_scores, 25, method='midpoint')
            Q3 = np.percentile(betweenness_centrality_scores, 75, method='midpoint')
            IQR = Q3 - Q1
            outlier_threshold = round(Q3 + 1.5 * IQR, 2)

            # Iterate over nodes and determine if each betweenness centrality score exceeds the outlier threshold
            hubs_list = []
            [hubs_list.append(x) for x in betweenness_centrality.keys() if
             betweenness_centrality[x] > outlier_threshold]
            hub_dict = {i: 1 if i in list(set(hubs_list) & set(self._node_labels)) else 0 for i in self._node_labels}
            if return_type == 'dict':
                return hub_dict
            if return_type == 'list':
                return list(hub_dict.values())

        def get_betweenness_centrality(self, graph=None, return_type='list'):
            """
            Calculate and return the betweenness centrality scores for all nodes in the network.

            Betweenness centrality is a measure of the importance of a node in a network based on the number of shortest paths
            that pass through that node. Higher betweenness centrality indicates nodes that act as crucial bridges or connectors
            within the network.

            :param graph: networkx.Graph, optional
                The graph for which to calculate betweenness centrality scores. If not provided, the CaGraph object's graph is used.
            :param return_type: str, optional
                The format in which to return the betweenness centrality scores. 'list' returns a list of centrality scores,
                'dict' returns a dictionary with nodes as keys and their centrality scores as values.

            :return: list or dict
                The betweenness centrality scores for nodes in the network, either as a list or a dictionary, based on the
                specified 'return_type'.
            """
            if graph is None:
                graph = self._graph
            # Calculate betweenness centrality
            if return_type == 'dict':
                return nx.betweenness_centrality(graph, normalized=True, endpoints=False)
            elif return_type == 'list':
                return list(nx.betweenness_centrality(graph, normalized=True, endpoints=False).values())

        def get_connected_components(self, graph=None) -> list:
            """
            Retrieve connected components in the network, excluding isolated nodes.

            This method identifies and returns connected components in the network graph. Connected components are groups of nodes
            that are connected to each other, and the method filters out isolated nodes (components with only one node) from the
            result.

            :param graph: networkx.Graph, optional
                The graph for which to find connected components. If not provided, the CaGraph object's graph is used.

            :return: List[List[int]]
                A list of connected components, where each component is represented as a list of node labels (as integers).
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
            Retrieve and return the largest connected component of the network.

            This method identifies and extracts the largest connected component from the network graph. The largest connected
            component is a subgraph containing the most extensive group of nodes that are connected to each other.

            :param graph: networkx.Graph, optional
                The graph from which to extract the largest connected component. If not provided, the CaGraph object's graph is used.

            :return: networkx.Graph
                A networkx.Graph object representing the largest connected component of the network.
            """
            if graph is None:
                graph = self._graph
            largest_component = max(nx.connected_components(graph), key=len)
            return graph.subgraph(largest_component)

        def get_clustering_coefficient(self, graph=None, return_type='list'):
            """
            Calculate and return the clustering coefficient for each node in the network.

            The clustering coefficient of a node is a measure of the degree to which its neighbors are also connected to each other.
            Higher clustering coefficients indicate nodes that are part of densely interconnected local subgraphs.

            :param graph: networkx.Graph, optional
                The graph for which to calculate clustering coefficients. If not provided, the CaGraph object's graph is used.
            :param return_type: str, optional
                The format in which to return clustering coefficients. 'list' returns a list of coefficients,
                'dict' returns a dictionary with nodes as keys and their clustering coefficients as values.

            :return: list or dict
                The clustering coefficients for nodes in the network, either as a list or a dictionary, based on the specified 'return_type'.
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
            Calculate and return the degree of each node in the network.

            The degree of a node in a network is the number of edges incident to that node, which represents how many
            connections or neighbors a node has in the network.

            :param graph: networkx.Graph, optional
                The graph for which to calculate node degrees. If not provided, the CaGraph object's graph is used.
            :param return_type: str, optional
                The format in which to return node degrees. 'list' returns a list of degrees,
                'dict' returns a dictionary with nodes as keys and their degrees as values.

            :return: list or dict
                The node degrees for nodes in the network, either as a list or a dictionary, based on the specified 'return_type'.
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
               Calculate and return the correlated pair ratio for each node in the network.

               The correlated pair ratio is a measure of the number of connections each neuron has in the network, divided by the
               total number of neurons in the network. It provides insight into the relative connectivity of individual neurons
               within the larger network.

               This method is described in Jimenez et al. 2020 (https://www.nature.com/articles/s41467-020-17270-w#Sec8).

               :param graph: networkx.Graph, optional
                   The graph for which to calculate correlated pair ratios. If not provided, the CaGraph object's graph is used.
               :param return_type: str, optional
                   The format in which to return correlated pair ratios. 'list' returns a list of ratios,
                   'dict' returns a dictionary with nodes as keys and their ratios as values.

               :return: list or dict
                   The correlated pair ratios for nodes in the network, either as a list or a dictionary, based on the specified 'return_type'.
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

        def get_eigenvector_centrality(self, graph=None, return_type='list'):
            """
             Calculate and return the eigenvector centrality for each node in the network.

             Eigenvector centrality is a measure of the influence that each node has on the entire network, taking into account
             its connections to other influential nodes. Nodes with higher eigenvector centrality are well-connected to other
             nodes with high centrality, indicating their importance in the network.

             :param graph: networkx.Graph, optional
                 The graph for which to calculate eigenvector centrality. If not provided, the CaGraph object's graph is used.
             :param return_type: str, optional
                 The format in which to return eigenvector centrality values. 'list' returns a list of centrality values,
                 'dict' returns a dictionary with nodes as keys and their centrality values as values.

             :return: list or dict
                 The eigenvector centrality values for nodes in the network, either as a list or a dictionary, based on the
                 specified 'return_type'.
             """
            if graph is None:
                graph = self._graph
            eigenvector_centrality = nx.eigenvector_centrality(graph, max_iter=500)
            if return_type == 'dict':
                return eigenvector_centrality
            if return_type == 'list':
                return list(eigenvector_centrality.values())

        def get_communities(self, graph=None, return_type='dict', **kwargs):
            """
            Detect and return communities within the network.

            Communities are groups of nodes that exhibit higher connectivity and interactions among themselves than with nodes outside
            the community. This method employs the modularity optimization algorithm to find these communities.

            :param graph: networkx.Graph, optional
                The graph for which to detect communities. If not provided, the CaGraph object's graph is used.
            :param return_type: str, optional
                The format in which to return community assignments. 'list' returns a list of community IDs for each node,
                'dict' returns a dictionary with nodes as keys and their corresponding community IDs as values.

            :return: list or dict
                The community assignments for nodes in the network, either as a list of community IDs or a dictionary, based on
                the specified 'return_type'.
            """
            if graph is None:
                graph = self._graph
            communities = list(nx.algorithms.community.greedy_modularity_communities(graph, **kwargs))
            sorted(communities)
            community_id = {}
            for i in range(len(communities)):
                for j in list(communities[i]):
                    community_id[j] = i
            if return_type == 'dict':
                return community_id
            # Todo: fix this so that returning list returns each node in the correct index
            # if return_type == 'list':
            #     return list(community_id.values())

        # Todo: getting stuck on small world analysis when computing sigma -- infinite loop may be due to computing the average clustering coefficient or the average shortest path length -- test
        def get_smallworld_largest_subnetwork(self, graph=None) -> float:
            """
            Calculate and return the small-worldness index for the largest connected subnetwork within the network.

            The small-worldness index measures the degree to which a network exhibits small-world properties, such as high
            clustering and short path lengths. It is calculated using the sigma metric from networkx's small-world
            functionality.

            :param graph: networkx.Graph, optional
                The graph for which to calculate the small-worldness index. If not provided, the largest connected subnetwork
                of the CaGraph object's graph is used.

            :return: float
                The small-worldness index for the network. If the largest subnetwork has less than four nodes, a RuntimeError
                is raised as sigma cannot be computed.
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

        # Todo: make more functional
        # Todo: make naming conventions consistent/ best practice
        def compare_graphs(self, graph1, graph2):
            """
            Compare two graphs and return their graph edit distance.

            The graph edit distance is a measure of the dissimilarity between two graphs. It quantifies the minimum number of
            operations (such as adding, deleting, or modifying nodes and edges) required to transform one graph into the other.

            :param graph1: networkx.Graph
                The first graph to be compared.
            :param graph2: networkx.Graph
                The second graph to be compared.

            :return: int
                The graph edit distance between the two graphs, indicating their dissimilarity.
            """
            return nx.graph_edit_distance(graph1, graph2)

    class Plotting:
        def __init__(self, neuron_dynamics, time, correlation_matrix, graph, num_neurons):
            self._num_neurons = num_neurons
            self._time = time
            self._neuron_dynamics = neuron_dynamics
            self._correlation_matrix = correlation_matrix
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
        def correlation_matrix(self):
            return self._correlation_matrix

        def plot_correlation_heatmap(self, correlation_matrix=None, title=None, y_label=None, x_label=None,
                                     show_plot=True,
                                     save_plot=False, save_path=None, dpi=300, save_format='png'):
            """
                Plot a heatmap of the correlation matrix.

                This method generates a heatmap to visualize the correlation matrix, which represents pairwise correlations
                between neurons. The heatmap is color-coded to highlight the strength of correlations.

                :param correlation_matrix: numpy.ndarray, optional
                    The correlation matrix to be visualized. If not provided, the correlation matrix of the CaGraph
                    object is used.
                :param title: str, optional
                    Title for the heatmap plot.
                :param y_label: str, optional
                    Label for the y-axis.
                :param x_label: str, optional
                    Label for the x-axis.
                :param show_plot: bool, optional
                    Whether to display the heatmap plot. Default is True.
                :param save_plot: bool, optional
                    Whether to save the plot as an image file.
                :param save_path: str, optional
                    Path to save the plot image. If not provided, the current working directory is used.
                :param dpi: int, optional
                    Dots per inch for the saved image.
                :param save_format: str, optional
                    Format of the saved image, e.g., 'png', 'jpg'.

                :return: None
                """
            if correlation_matrix is None:
                correlation_matrix = self._correlation_matrix()
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
                elif not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                plt.savefig(fname=save_path, dpi=dpi, format=save_format)

        def get_single_neuron_timecourse(self, neuron_trace_number) -> np.ndarray:
            """
                Return the time vector stacked on the recorded calcium fluorescence for the specified neuron.

                This method extracts the time vector and calcium fluorescence data for a specific neuron, stacks them together,
                and returns the combined time course.

                :param neuron_trace_number: int
                    The index of the neuron's trace to be retrieved.

                :return: numpy.ndarray
                    An array containing the time vector stacked on the calcium fluorescence data for the specified neuron.
                """
            neuron_timecourse_selection = neuron_trace_number
            return np.vstack((self._time, self._neuron_dynamics[neuron_timecourse_selection, :]))

        def plot_single_neuron_timecourse(self, neuron_trace_number, title=None, y_label=None, x_label=None,
                                          show_plot=True,
                                          save_plot=False, save_path=None, dpi=300, save_format='png'):
            """
            Plots the time course of a single neuron's calcium fluorescence data.

            This method generates a line plot of the calcium fluorescence data for a single neuron over time. You can customize
            the plot with optional parameters.

            :param save_format: str
                The format in which to save the plot (e.g., 'png', 'jpg', 'svg').
            :param neuron_trace_number: int
                The index of the neuron's trace to be plotted.
            :param title: str, optional
                The title of the plot.
            :param y_label: str, optional
                The label for the y-axis.
            :param x_label: str, optional
                The label for the x-axis.
            :param show_plot: bool, optional
                Whether to display the plot (default is True).
            :param save_plot: bool, optional
                Whether to save the plot to a file (default is False).
            :param save_path: str, optional
                The directory where the plot should be saved (default is the current working directory).
            :param dpi: int, optional
                The resolution of the saved plot in dots per inch (default is 300).

            :return: None
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
                elif not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                plt.savefig(fname=save_path, dpi=dpi, format=save_format)

        def plot_multi_neuron_timecourse(self, neuron_trace_labels, palette=None, title=None, y_label=None,
                                         x_label=None,
                                         show_plot=True, save_plot=False, save_path=None, dpi=300, save_format='png'):
            """
                Plots multiple individual calcium fluorescence traces stacked vertically.

                This method generates a multi-panel plot, where each panel represents the calcium fluorescence data of an individual
                neuron. The traces are stacked vertically for visual comparison.

                :param save_format: str
                    The format in which to save the plot (e.g., 'png', 'jpg', 'svg').
                :param neuron_trace_labels: list
                    A list of neuron indices to be plotted. Each index corresponds to an individual neuron's trace.
                :param palette: list, optional
                    A list of colors to use for plotting the traces. If not provided, a default color palette is used.
                :param title: str, optional
                    The title of the plot.
                :param y_label: str, optional
                    The label for the y-axis.
                :param x_label: str, optional
                    The label for the x-axis.
                :param show_plot: bool, optional
                    Whether to display the plot (default is True).
                :param save_plot: bool, optional
                    Whether to save the plot to a file (default is False).
                :param save_path: str, optional
                    The directory where the plot should be saved (default is the current working directory).
                :param dpi: int, optional
                    The resolution of the saved plot in dots per inch (default is 300).

                :return: None
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
                elif not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                plt.savefig(fname=save_path, dpi=dpi, format=save_format)

        def plot_all_neurons_timecourse(self, title=None, y_label=None, x_label=None, show_plot=True, save_plot=False,
                                        save_path=None, dpi=300, save_format='png'):
            """
                Plots the calcium fluorescence timecourses of all neurons in the dataset.

                This method generates a plot where each neuron's calcium fluorescence timecourse is displayed. Neuron traces are
                plotted sequentially, and they can be overlaid or shown separately, depending on the dataset size.

                :param save_format: str
                    The format in which to save the plot (e.g., 'png', 'jpg', 'svg').
                :param title: str, optional
                    The title of the plot.
                :param y_label: str, optional
                    The label for the y-axis.
                :param x_label: str, optional
                    The label for the x-axis.
                :param show_plot: bool, optional
                    Whether to display the plot (default is True).
                :param save_plot: bool, optional
                    Whether to save the plot to a file (default is False).
                :param save_path: str, optional
                    The directory where the plot should be saved (default is the current working directory).
                :param dpi: int, optional
                    The resolution of the saved plot in dots per inch (default is 300).

                :return: None
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
                elif not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                plt.savefig(fname=save_path, dpi=dpi, format=save_format)

    class Visualization:
        def __init__(self, cagraph_obj):
            """
            Initializes a Visualization object.

            :param cagraph_obj: CaGraph
                The CaGraph object to be visualized.
            """
            self.cagraph_obj = cagraph_obj
            pass

        def _interactive_network_input_validator(self, input_object):
            """
            Validate the input object.

            :param input_object: CaGraph
                The input object to be validated.
            :return: CaGraph
            """
            if isinstance(input_object, CaGraph):
                return input_object
            else:
                raise TypeError('cagraph_obj must be type cagraph.CaGraph.')

        def show_interactive_network(self, **kwargs):
            """
            Display an interactive visualization of the CaGraph object.

            :param kwargs:
                Additional keyword arguments to customize the visualization.
            """
            self._interactive_network_input_validator(input_object=self.cagraph_obj)
            visualization.interactive_network(cagraph_obj=self.cagraph_obj, **kwargs)


# %% Time samples
class CaGraphTimeSamples:
    """
                A class for running time-sample analyses on a single dataset.

                This class allows you to perform time-sample analyses on a single dataset, creating CaGraph objects for different time
                samples under various conditions.

                :param data: numpy.ndarray
                    The dataset containing neural activity data. Rows represent neurons, columns represent time points.
                :param time_samples: list of tuples
                    A list of time sample intervals for different conditions. Each tuple should contain two integers, representing
                    the start and end time points of a time sample.
                :param condition_labels: list of str
                    Labels for different conditions, corresponding to the time samples.
                :param node_labels: list of int
                    Labels for individual nodes (neurons). If not provided, default labels are generated.
                :param node_metadata: dict
                    Metadata associated with nodes.
                :param dataset_id: str
                    A unique identifier for the dataset.
                :param threshold: float
                    A threshold for graph edge creation based on correlation. If not provided, it will be generated.

                Attributes:
                data (numpy.ndarray): The dataset containing neural activity data.
                dt (float): The time interval between data points.
                num_neurons (int): The number of neurons in the dataset.
                data_id (str): A unique identifier for the dataset.
                node_labels (list): Labels for individual nodes (neurons).
                threshold (float): The threshold for graph edge creation based on correlation.
                condition_identifiers (list): Labels for different conditions, corresponding to the time samples.
            """

    def __init__(self, data, time_samples, condition_labels, correlation_method='pearson', node_labels=None, node_metadata=None,
                 dataset_id=None, threshold=None, **correlation_kwargs):
        """
                Initializes a CaGraphTimeSamples object.

                :param data: numpy.ndarray
                    The dataset containing neural activity data.
                :param time_samples: list of tuples
                    A list of time sample intervals for different conditions.
                :param condition_labels: list of str
                    Labels for different conditions.
                :param node_labels: list of int
                    Labels for individual nodes (neurons).
                :param node_metadata: dict
                    Metadata associated with nodes.
                :param dataset_id: str
                    A unique identifier for the dataset.
                :param threshold: float
                    A threshold for graph edge creation based on correlation.
                """

        # Check that the input data is in the correct format and load dataset
        self.__input_validator(data=data, time_samples=time_samples, condition_labels=condition_labels)

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
            self._threshold = self.__generate_threshold(correlation_method=correlation_method, **correlation_kwargs)

        # Generate node labels
        if node_labels is None:
            self._node_labels = np.linspace(0, np.shape(self.data)[0] - 2,
                                            np.shape(self.data)[0] - 1).astype(int)
        else:
            self._node_labels = node_labels

        # Add a series of private attributes which are CaGraph objects
        for i, sample in enumerate(time_samples):
            setattr(self, f'__{condition_labels[i]}_cagraph',
                    CaGraph(data=self._data[:, sample[0]:sample[1]], correlation_method = correlation_method, node_labels=self._node_labels,
                            node_metadata=node_metadata, threshold=self._threshold), **correlation_kwargs)

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

    def __input_validator(self, data, time_samples, condition_labels):
        """
        Validates and loads input data for the CaGraphTimeSamples class.

        This method performs validation on the input data to ensure it is in the correct format. It supports loading data
        either from a numpy.ndarray or from specific file formats such as .csv or .nwb. The data is then stored in the
        `_data` attribute.

        :param data: numpy.ndarray or str
            The input data, which can be a numpy.ndarray containing neural activity data, a file path to a .csv or .nwb file
            for data loading, or a numpy.ndarray containing time samples.
        :param time_samples: list of tuples
            A list of time sample intervals for different conditions.
        :param condition_labels: list of str
            Labels for different conditions.

        Raises:
            TypeError: If data is not a supported data format or file type.
            ValueError: If the number of time_samples does not match the number of condition_labels.
        """
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

        if len(time_samples) != len(condition_labels):
            raise ValueError(
                'The number of time_samples provided does not match the number of condition_labels provided.')

    def __generate_threshold(self, correlation_method='pearson', **correlation_kwargs) -> float:
        """
        Generates a threshold for the provided dataset as described in the preprocess module.
        This threshold generation will use the full dataset.

        Returns the calculated threshold as a float.

        :return: float
        """
        return preprocess.generate_average_threshold(data=self.data[1:, :], shuffle_iterations=10, correlation_method=correlation_method, **correlation_kwargs)

    # Public utility methods
    def save(self, file_path=None):
        """
        Save the CaGraphTimeSamples object to a binary file using pickle.

        :param file_path: str, optional
            The path to the file where the object will be saved. If not provided, the default filename 'obj.cagraph' will be used.
        :return: None
        """
        if file_path is None:
            file_path = 'obj.cagraph'
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path):
        """
        Load a CaGraphTimeSamples object from a binary file using pickle.

        :param file_path: str
            The path to the file containing the saved CaGraphTimeSamples object.
        :return: CaGraphTimeSamples
            The loaded CaGraphTimeSamples object.
        """
        with open(file_path, 'rb') as file:
            cagraphtimesamples_obj = pickle.load(file)
        return cagraphtimesamples_obj

    def get_cagraph(self, condition_label):
        """
        Get the CaGraph object associated with the specified condition label.

        :param condition_label: str
            The label identifying the condition for which the CaGraph object is requested.
        :return: CaGraph
            The CaGraph object corresponding to the provided condition label.
        """
        return getattr(self, f'__{condition_label}_cagraph')

    def get_full_report(self, analysis_selections=None, save_report=False, save_path=None, save_filename=None,
                        save_filetype=None):
        """
            Generate a consolidated report that combines individual condition reports into a single report.

            This method combines the reports for all conditions included in the CaGraphTimeSamples object
            and produces a comprehensive report in a tabular structure, where each column corresponds to
            an analysis from one of the individual condition reports.

            :param save_report: bool
                Whether to save the generated report to a file.
            :param save_path: str, optional
                The directory path where the report file should be saved. If not provided, the current working
                directory will be used.
            :param save_filename: str, optional
                The base filename for the saved report file. If not provided, it defaults to 'report'.
            :param save_filetype: str, optional
                The format in which to save the report. Options include 'csv', 'HDF5', and 'xlsx'.
                If not provided, the default is 'csv'.
            :return: pandas.DataFrame
                A DataFrame containing the consolidated report.
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

        if analysis_selections is not None:
            # Todo: Check that analysis_selections are valid
            selections_str = '|'.join(analysis_selections)
            full_report_df = full_report_df.filter(regex=selections_str)

        # Save the report
        if save_report:
            if save_filename is None:
                save_filename = 'report'
            if save_path is None:
                save_path = os.getcwd() + '/'
            elif not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
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
        Class for running batched analyses on multiple datasets.

        CaGraphBatch allows you to analyze multiple datasets simultaneously. You can provide a list of dataset paths or
        specify a directory containing dataset files. Additionally, you can set a group identifier and specify the
        threshold for analysis.

        :param data: str or list
            If 'data' is a string, it should be the path to a directory containing dataset files (e.g., CSV or NWB).
            If 'data' is a list, it should contain paths to individual dataset files.
        :param dataset_labels: list, optional
            Labels for individual datasets. Use this when providing a list of datasets. The number of labels must match
            the number of datasets.
        :param group_id: str, optional
            An identifier for the group of datasets.
        :param threshold: float, optional
            The threshold for data analysis. If not provided, the threshold will be determined automatically for each dataset.
        :param threshold_averaged: bool, optional
            If True, an average threshold will be computed across all datasets to ensure consistent analysis thresholds.
        """

    def __init__(self, data, correlation_method = 'pearson', dataset_labels=None, group_id=None, threshold=None, threshold_averaged=False, **correlation_kwargs):
        """
            Initialize a CaGraphBatch object for batched analyses on multiple datasets.

            Parameters:
            :param data: str or list
                If 'data' is a string, it should be the path to a directory containing dataset files (e.g., CSV or NWB).
                If 'data' is a list, it should contain paths to individual dataset files.
            :param dataset_labels: list, optional
                Labels for individual datasets. Use this when providing a list of datasets. The number of labels must match
                the number of datasets.
            :param group_id: str, optional
                An identifier for the group of datasets.
            :param threshold: float, optional
                The threshold for data analysis. If not provided, the threshold will be determined automatically for each dataset.
            :param threshold_averaged: bool, optional
                If True, an average threshold will be computed across all datasets to ensure consistent analysis thresholds.

            The 'data' parameter must be specified to load the dataset. Depending on the data format and input, it can be a
            directory or a list of dataset file paths. Dataset labels, group identifiers, and thresholds can also be specified.
            """
        # Check that the input data is in the correct format and load dataset
        self.__input_validator(data=data)

        # Todo: Optimize following logic
        if type(data) == list:
            if dataset_labels is not None:
                if len(dataset_labels) != len(data):
                    raise ValueError("The number of dataset_labels must match the number or datasets in 'data'.")
            else:
                raise ValueError("The dataset_labels input must be specified if a list of data is provided.")
            data_list = data
            data_path = ''
        elif type(data) == str:
            files = os.listdir(data)
            data_list = []
            for file in files:
                if file.endswith('.csv') or file.endswith('.nwb'):
                    data_list.append(file)
            dataset_labels = []
            for idx, dataset in enumerate(data_list):
                dataset_labels.append(dataset[:-4])
            data_path = data

        # Set threshold
        if threshold is not None:
            self._threshold = threshold

        # Todo: check if error when using threshold_averaged
        elif threshold_averaged:
            self._threshold = self.__generate_averaged_threshold(data_path=data_path, correlation_method=correlation_method, dataset_keys=data_list, **correlation_kwargs)
        else:
            self._threshold = None
            self._batch_threshold = None
        if group_id is not None:
            self._group_id = group_id

        # Construct CaGraph objects for each dataset
        self._dataset_identifiers = []
        for idx, dataset in enumerate(data_list):
            if isinstance(dataset, str):
                data = np.genfromtxt(data_path + dataset, delimiter=",")
            elif isinstance(dataset, np.ndarray):
                data = dataset
            try:
                setattr(self, f'__{dataset_labels[idx]}_cagraph', CaGraph(data=data, correlation_method=correlation_method, threshold=self._threshold, **correlation_kwargs))
                # Add a series of private attributes which are CaGraph objects
                self._dataset_identifiers.append(dataset_labels[idx])
            except Exception as e:
                print(f"Exception occurred for dataset {dataset_labels[idx]}: " + repr(e))

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

    def __input_validator(self, data):
        """
        Validate the input data to ensure it meets the requirements for creating CaGraphBatch objects.

        Parameters:
        :param data: str or list
            The data to be validated. It can be either a list of dataset paths or a directory path containing dataset files.

        Raises:
        - TypeError: If the input data does not meet the required format.
        - ValueError: If the provided directory path does not exist.

        This method checks the input data to ensure it is in the correct format. For a list of datasets, it verifies that each
        dataset meets the qualifications for CaGraph objects. If 'data' is a directory path, it checks if the directory exists.
        """
        if type(data) == list:
            # For each dataset in the list check that each meets the input qualifications for CaGraph objects
            for dataset in data:
                if isinstance(dataset, np.ndarray):
                    pass
                elif isinstance(dataset, str):
                    if not dataset.endswith('csv') and not dataset.endswith('nwb'):
                        raise TypeError('File path to each dataset must have a .csv or .nwb file to load.')
                else:
                    raise TypeError('Data must be passed as a str containing a .csv or .nwb file, or as numpy.ndarray.')
        elif type(data) == str and not os.path.exists(os.path.dirname(data)):
            raise ValueError('Path provided for data parameter does not exist.')

    # Todo: Urgent -- make sure this is compatible with datasets passed as NumPy arrays
    # Todo: currently, user must specify the dataset_keys as .csv in the string - make more flexible
    def __generate_averaged_threshold(self, data_path, dataset_keys, correlation_method='pearson', **correlation_kwargs):
        """
        Compute an averaged threshold by calculating the mean of the recommended thresholds for each individual dataset.

        Parameters:
        :param data_path: str
            The path to the directory containing the dataset files.

        :param dataset_keys: list of str
            A list of dataset keys (filenames) for which thresholds will be calculated.

        Returns:
        - float
            The averaged threshold computed as the mean of individual dataset thresholds.

        This method computes an averaged threshold by iterating over the provided dataset keys, loading each dataset from the
        specified data path, and calculating the recommended threshold for each dataset using the 'generate_average_threshold'
        function from the 'preprocess' module. It then returns the mean of these individual thresholds as the averaged threshold.
        """
        store_thresholds = []
        for dataset in dataset_keys:
            data = np.genfromtxt(f'{data_path}{dataset}', delimiter=",")
            store_thresholds.append(preprocess.generate_average_threshold(data=data[1:, :], correlation_method=correlation_method, shuffle_iterations=10, **correlation_kwargs))
        return np.mean(store_thresholds)

    # Public utility methods
    def save(self, file_path=None):
        """
        Save the CaGraphBatch instance to a binary file using pickle.

        Parameters:
        :param file_path (str, optional):
            The path to the file where the CaGraphBatch instance will be saved. If not provided, the default filename 'obj.cagraph' is used.

        This method allows you to save the current CaGraphBatch instance to a binary file using the 'pickle' module, preserving the object's state for future use.
        """
        if file_path is None:
            file_path = 'obj.cagraph'
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path):
        """
        Load a CaGraphBatch instance from a saved binary file.

        Parameters:
        :param file_path (str):
            The path to the binary file containing the saved CaGraphBatch instance.

        Returns:
        - CaGraphBatch:
            A new instance of the CaGraphBatch class loaded from the specified file.

        This method allows you to load a previously saved CaGraphBatch instance from a binary file created using the 'save' method.
        """
        with open(file_path, 'rb') as file:
            cagraphbatch_obj = pickle.load(file)
        return cagraphbatch_obj

    def get_cagraph(self, condition_label) -> CaGraph:
        """
        Retrieve a CaGraph object for the specified dataset condition_label.

        Parameters:
        :param condition_label (str):
            The label or identifier of the dataset condition for which you want to retrieve the CaGraph object.

        Returns:
        - CaGraph:
            The CaGraph object associated with the specified condition_label.

        This method allows you to access the CaGraph object associated with a specific dataset condition_label within the CaGraphBatch instance.
        """
        return getattr(self, f'__{condition_label}_cagraph')

    def get_full_report(self, analysis_selections=None, save_report=False, save_path=None, save_filename=None,
                        save_filetype=None):
        """
            Generate a consolidated report for all datasets in the batched sample, combining the results of base analyses
            included in the CaGraph object's get_report() method. This report is presented as a pandas DataFrame and can be
            optionally saved to a file.

            Parameters:
            :param save_report (bool, optional):
                If True, the report will be saved to a file. Defaults to False.

            :param save_path (str, optional):
                The directory path where the report file will be saved. If not specified, the current working directory is used.

            :param save_filename (str, optional):
                The name of the report file (excluding file extension). Defaults to 'report'.

            :param save_filetype (str, optional):
                The file format in which to save the report. Available formats: 'csv', 'HDF5', 'xlsx'. Defaults to 'csv'.

            Returns:
            - pd.DataFrame:
                A pandas DataFrame containing the consolidated report for all datasets.

            This method creates a summary report that combines the results of base analyses from individual datasets in the batch.
            The columns in the report correspond to specific analyses, and the rows correspond to different datasets.
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

        if analysis_selections is not None:
            # Todo: Check that analysis_selections are valid
            selections_str = '|'.join(analysis_selections)
            full_report_df = full_report_df.filter(regex=selections_str)

        # Save the report
        if save_report:
            if save_filename is None:
                save_filename = 'report'
            if save_path is None:
                save_path = os.getcwd() + '/'
            elif not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            if save_filetype is None or save_filetype == 'csv':
                full_report_df.to_csv(save_path + save_filename + '.csv', index=True)
            elif save_filetype == 'HDF5':
                full_report_df.to_hdf(save_path + save_filename + '.h5', key=save_filename, mode='w')
            elif save_filetype == 'xlsx':
                full_report_df.to_excel(save_path + save_filename + 'xlsx', index=True)
        return full_report_df

    # Todo: Allow individual datasets to be parsed
    def save_individual_dataset_reports(self, save_path=None, save_filetype=None):
        """
        Save individual reports for each of the specified datasets.

        This method iterates through all datasets in the batched sample, generates a report for each dataset, and saves
        them individually. Each dataset's report is saved in a file with a filename derived from the dataset's name.

        Parameters:
        :param save_path (str, optional):
            The directory path where the individual reports will be saved. If not specified, the current working directory is used.

        :param save_filetype (str, optional):
            The file format in which to save the individual reports. Available formats: 'csv', 'HDF5', 'xlsx'.
        """
        # Iterate through all datasets and save report for each
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        for key in self.dataset_identifiers:
            cagraph_obj = self.get_cagraph(key)
            cagraph_obj.get_report(save_report=True, save_path=save_path, save_filename=key + '_report',
                                   save_filetype=save_filetype)


# %% Batched and sampled analyses
class CaGraphBatchTimeSamples:
    """
    Class for running batched analyses on datasets that have distinct time periods to
    separate into samples.

    Node metadata cannot be added to CaGraph objects in the batched
    analysis. Future versions will include the node_metadata attribute.
    """

    def __init__(self, data, time_samples, condition_labels, correlation_method='pearson', dataset_labels=None, group_id=None, threshold=None,
                 threshold_averaged=False, **correlation_kwargs):
        """
                Initialize a CaGraphBatchTimeSamples object.

                Parameters:
                :param data (str or list of str or list of np.ndarray): The path to data or a list of paths to data. A list of file
                  paths can be provided if the data is split into multiple files. Each file represents a dataset. If loading from
                  CSV or NWB files, the data should be a comma-separated values (CSV) file or a Neurodata Without Borders (NWB)
                  file. If using NumPy arrays, data can be a list of arrays or a single array. Data will be loaded from these sources.
                  Alternatively, a list of directories can be provided, and all CSV and NWB files within those directories will be
                  treated as datasets. In this case, dataset_labels should be provided to specify the names for the datasets.
                  File paths should be in string format.

                :param time_samples (list of tuples): A list of time sample periods to separate the data into distinct samples.
                  Each tuple represents a time period and should have the format (start, end), where 'start' is the starting time
                  and 'end' is the ending time for a sample.

                :param condition_labels (list of str): A list of condition labels corresponding to the time samples. Each label describes
                  the condition or event during the respective time period. The number of condition labels should match the number
                  of time samples.

                :param dataset_labels (list of str, optional): A list of labels to identify each dataset when using directories. The number
                  of dataset labels should match the number of datasets.

                :param group_id (str, optional): An optional identifier for a group or batch of datasets.

                :param threshold (float, optional): The threshold for identifying edges in the connectivity graph. You can manually set
                 this threshold value. If not specified, a threshold will be computed for each dataset separately.

                :param threshold_averaged (bool, optional): If set to True, an averaged threshold will be computed based on the
                  threshold values for each dataset. This averaged threshold is applied consistently to all datasets.

                Note: Dataset labels are used to identify individual datasets when analyzing and reporting.


        """
        # Check that the input data is in the correct format and load dataset
        self.__input_validator(data=data, time_samples=time_samples, condition_labels=condition_labels)

        # Todo: check logic for optimizations
        if type(data) == list:
            if dataset_labels is not None:
                if len(dataset_labels) != len(data):
                    raise ValueError("The number of dataset_labels must match the number or datasets in 'data'.")
            else:
                raise ValueError("The dataset_labels input must be specified if a list of data is provided.")
            data_list = data
            data_path = ''
        elif type(data) == str:
            files = os.listdir(data)
            data_list = []
            for file in files:
                if file.endswith('.csv') or file.endswith('.nwb'):
                    data_list.append(file)
            dataset_labels = []
            for idx, dataset in enumerate(data_list):
                dataset_labels.append(dataset[:-4])
            data_path = data

        # Set threshold
        if threshold is not None:
            self._threshold = threshold
        elif threshold_averaged:
            self._threshold = self.__generate_averaged_threshold(data_list=data_list, correlation_method=correlation_method, data_path=data_path, **correlation_kwargs)
        else:
            self._threshold = None
            self._batch_threshold = None
        if group_id is not None:
            self._group_id = group_id

        # Construct CaGraph objects for each dataset
        self._dataset_identifiers = []
        for idx, dataset in enumerate(data_list):
            if isinstance(dataset, str):
                data = np.genfromtxt(data_path + dataset, delimiter=",")
            elif isinstance(dataset, np.ndarray):
                data = dataset
            try:
                if hasattr(self,
                           '_batch_threshold'):  # If the _batch_threshold attribute exists, the threshold should be set for each dataset
                    self._threshold = self.__generate_threshold(data=data,correlation_method=correlation_method, **correlation_kwargs)
                    # Add a series of private attributes which are CaGraph objects
                for i, sample in enumerate(time_samples):
                    setattr(self, f'__{dataset_labels[idx]}_{condition_labels[i]}_cagraph',
                            CaGraph(data=data[:, sample[0]:sample[1]], threshold=self._threshold, correlation_method=correlation_method, **correlation_kwargs))
                    self._dataset_identifiers.append(dataset_labels[idx] + '_' + condition_labels[i])
            except Exception as e:
                print(f"Exception occurred for dataset {dataset_labels[idx]}: " + repr(e))

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

    def __input_validator(self, data, time_samples, condition_labels):
        """
        Performs input validation for CaGraphBatchTimeSamples class.

        Parameters:
        :param data: The input data, which can be a list of file paths (str), a list of NumPy arrays, or a directory path (str).
        :param time_samples: A list of time sample periods (tuples) to separate the data into distinct samples.
        :param condition_labels: A list of condition labels (str) corresponding to the time_samples.

        Raises:
        - TypeError: If the data does not meet the required format. It should be a list of file paths, a list of NumPy arrays,
          or a directory path, with file paths ending in '.csv' or '.nwb'.
        - ValueError: If the path provided for data does not exist or if the number of time_samples does not match the
          number of condition_labels.
        """
        if type(data) == list:
            # For each dataset in the list check that each meets the input qualifications for CaGraph objects
            for dataset in data:
                if isinstance(dataset, np.ndarray):
                    pass
                elif isinstance(dataset, str):
                    if not dataset.endswith('csv') and not dataset.endswith('nwb'):
                        raise TypeError('File path to each dataset must have a .csv or .nwb file to load.')
                else:
                    raise TypeError('Data must be passed as a str containing a .csv or .nwb file, or as numpy.ndarray.')
        elif type(data) == str and not os.path.exists(os.path.dirname(data)):
            raise ValueError('Path provided for data parameter does not exist.')

        if len(time_samples) != len(condition_labels):
            raise ValueError(
                'The number of time_samples provided does not match the number of condition_labels provided.')

    def __generate_threshold(self, data, correlation_method='pearson', **correlation_kwargs) -> float:
        """
        Generates a threshold for the provided dataset as described in the preprocess module.

        Parameters:
        :param data: A NumPy array containing calcium fluorescence data.

        Returns:
        float: The computed threshold value.

        The threshold is generated based on the provided calcium fluorescence data using the `preprocess.generate_average_threshold`
        function with a specified number of shuffle iterations.

        """
        return preprocess.generate_average_threshold(data=data, shuffle_iterations=10, correlation_method=correlation_method, **correlation_kwargs)

    def __generate_averaged_threshold(self, data_list, data_path=None, correlation_method='pearson', **correlation_kwargs):
        """
            Computes an averaged threshold by computing the mean of the recommended thresholds for each individual dataset.

            Parameters:
            :param data_list: A list of dataset file paths or NumPy arrays containing calcium fluorescence data.
            :param data_path: An optional path prefix for loading data from file paths.

            Returns:
            - float: The computed averaged threshold value.

            This method computes the mean threshold value by calculating recommended thresholds for each individual dataset
            in the 'data_list'. The thresholds are generated using the `preprocess.generate_average_threshold` function
            with a specified number of shuffle iterations.

            """
        store_thresholds = []
        for dataset in data_list:
            if data_path is not None and dataset.endswith('csv'):
                data = np.genfromtxt(f'{data_path}{dataset}.csv', delimiter=",")
            elif data_path is not None and dataset.endswith('nwb'):
                data = np.genfromtxt(f'{data_path}{dataset}.nwb', delimiter=",")
            elif isinstance(dataset, np.ndarray):
                data = dataset
            elif isinstance(dataset, str):
                data = np.genfromtxt(dataset, delimiter=",")
            store_thresholds.append(preprocess.generate_average_threshold(data=data[1:, :], shuffle_iterations=10, correlation_method=correlation_method, **correlation_kwargs))
        return np.mean(store_thresholds)

    # Public utility methods
    def save(self, file_path=None):
        """
        Save the CaGraphBatchTimeSamples object to a binary file using Pickle.

        Parameters:
        :param file_path (str, optional): The file path where the object should be saved. If not provided, a default file path
          with the extension '.cagraph' is used.

        Returns:
        None

        This method allows you to save the CaGraphBatchTimeSamples object to a binary file using Pickle. You can specify a
        'file_path' to save the object to a custom location or provide a different name, or it will use a default name if
        not provided.
        """
        if file_path is None:
            file_path = 'obj.cagraph'
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path):
        """
            Load a CaGraphBatchTimeSamples object from a binary file created with Pickle.

            Parameters:
            :param file_path (str): The path to the Pickle file containing the saved CaGraphBatchTimeSamples object.

            Returns:
            CaGraphBatchTimeSamples: The loaded CaGraphBatchTimeSamples object.

            This class method allows you to load a CaGraphBatchTimeSamples object from a binary file that was previously saved
            using the 'save' method. You should provide the 'file_path' to the Pickle file where the object is stored.
        """
        with open(file_path, 'rb') as file:
            cagraphbatchtimesamples_obj = pickle.load(file)
        return cagraphbatchtimesamples_obj

    def get_cagraph(self, condition_label) -> CaGraph:
        """
            Retrieve a CaGraph object for the specified condition label.

            Parameters:
            :param condition_label (str): The condition label associated with the desired CaGraph object.

            Returns:
            CaGraph: The CaGraph object for the specified condition label.

            This method allows you to obtain a CaGraph object from the CaGraphBatchTimeSamples instance based on the provided
            'condition_label'. The condition label should match one of the conditions associated with the datasets.
        """
        return getattr(self, f'__{condition_label}_cagraph')

    def get_full_report(self, analysis_selections=None, save_report=False, save_path=None, save_filename=None,
                        save_filetype=None):
        """
        Generate an organized report of all data in the batched sample.

        This method creates a comprehensive report of the data in the batched sample, including base analyses provided
        by the CaGraph object's `get_report()` method. The report is organized as a pandas DataFrame or a file (CSV, HDF5,
        or Excel) for convenient analysis and visualization.

        Parameters:
        :param save_report (bool, optional): Whether to save the report to a file. Default is False.
        :param save_path (str, optional): The directory where the report file should be saved. Default is None.
        :param save_filename (str, optional): The name of the report file. Default is None, which generates a default name.
        :param save_filetype (str, optional): The format of the report file (CSV, HDF5, or Excel). Default is None, which saves
          the report as a CSV file.
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

        if analysis_selections is not None:
            # Todo: Check that analysis_selections are valid
            selections_str = '|'.join(analysis_selections)
            full_report_df = full_report_df.filter(regex=selections_str)

        # Save the report
        if save_report:
            if save_filename is None:
                save_filename = 'report'
            if save_path is None:
                save_path = os.getcwd() + '/'
            elif not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            if save_filetype is None or save_filetype == 'csv':
                full_report_df.to_csv(save_path + save_filename + '.csv', index=True)
            elif save_filetype == 'HDF5':
                full_report_df.to_hdf(save_path + save_filename + '.h5', key=save_filename, mode='w')
            elif save_filetype == 'xlsx':
                full_report_df.to_excel(save_path + save_filename + 'xlsx', index=True)
        return full_report_df

    def save_individual_dataset_reports(self, save_path=None, save_filetype=None):
        """
        Save individual reports for each specified dataset.

        This method creates and saves individual reports for each dataset in the batch. The filenames for the reports are
        generated based on the dataset names from which the analyses are derived. These individual reports provide
        detailed analyses for each dataset, and the same analyses that can be obtained by creating a CaGraph object
        using a single dataset.

        Parameters:
        :param save_path (str, optional): The directory where the individual reports should be saved. Default is None.
        :param save_filetype (str, optional): The format of the report files (CSV, HDF5, or Excel). Default is None, which
          saves the reports as CSV files.
        """
        # Iterate through all datasets and save report for each
        for key in self.dataset_identifiers:
            cagraph_obj = self.get_cagraph(key)
            cagraph_obj.get_report(save_report=True, save_path=save_path, save_filename=key + '_report',
                                   save_filetype=save_filetype)


# %%  Matched analyses
# Todo: under development
# Todo: add option to include only matched cells or all cells - in the all-cell option, report on matched cells
class CaGraphMatched:
    """
        Class for running analyses on datasets that have been cell-tracked over time to identify the same cells.

        This class allows you to perform analyses on datasets that have been cell-tracked over time to identify matching cells
        between the datasets.

        Args:
        :param data (list): A list containing the input datasets for analysis. Each element in the list should be a NumPy array
          representing a dataset. These datasets should have been cell-tracked, so they have the same number of neurons
          in corresponding positions.
        :param dataset_labels (list): A list of labels for the datasets, one label for each dataset in the 'data' list.
        :param match_map (str): The path to a CSV file containing a mapping of cell indices between the datasets. The mapping
          should be represented as a CSV file where each row corresponds to a pair of matched cells, and each row contains
          two integers: the index of the cell in the first dataset and the index of the matching cell in the second dataset.
        :param matched_only (bool, optional): If True, only matched cells will be included in the analysis. If False, both matched
          and unmatched cells will be included. Default is True.
        :param threshold (float, optional): The threshold for edge detection in the analysis. Default is None, and a threshold
          will be automatically generated.

        Note:
        - The 'data' list should contain datasets in the same order as specified in 'dataset_labels'.
        - The 'match_map' file should represent matching indices between corresponding datasets.

        Example:
        ```python
        data = [dataset_0, dataset_1]
        labels = ["dataset_0", "dataset_1"]
        map_file = "cell_matching_map.csv"
        cagraph_matched = CaGraphMatched(data=data, dataset_labels=labels, match_map=map_file)
        ```

        This example creates a CaGraphMatched object with two datasets, labeled "dataset_0" and "dataset_1," and a cell
        matching map provided in the "cell_matching_map.csv" file.
        """

    def __init__(self, data, dataset_labels, match_map, correlation_method='pearson', matched_only=True, threshold=None, **correlation_kwargs):
        """
        Class for running analyses on datasets that have been cell-tracked over time to identify the same cells.

        Args:
        :param data (list): A list containing the input datasets for analysis. Each element in the list should be a NumPy array
          representing a dataset. These datasets should have been cell-tracked, so they have the same number of neurons
          in corresponding positions.
        :param dataset_labels (list): A list of labels for the datasets, one label for each dataset in the 'data' list.
        :param match_map (str): The path to a CSV file containing a mapping of cell indices between the datasets. The mapping
          should be represented as a CSV file where each row corresponds to a pair of matched cells, and each row contains
          two integers: the index of the cell in the first dataset and the index of the matching cell in the second dataset.
        :param matched_only (bool, optional): If True, only matched cells will be included in the analysis. If False, both matched
          and unmatched cells will be included. Default is True.
        :param threshold (float, optional): The threshold for edge detection in the analysis. Default is None, and a threshold
          will be automatically generated.

        """
        # Check that the input data is in the correct format and load dataset
        self.__input_validator(data=data, dataset_labels=dataset_labels)

        self._dataset_identifiers = dataset_labels

        # Load the cell matching indices map
        self._map = np.loadtxt(match_map, delimiter=',').astype(int)

        # Compute time interval and number of neurons
        self._dt = self._data_1[0, 1] - self._data_1[0, 0]

        if threshold is not None:
            self._threshold = threshold
        else:
            self._threshold = self.__generate_threshold(correlation_method=correlation_method, **correlation_kwargs)

        # Parse datasets using map
        dataset_0 = self._data_0[0, :]
        dataset_1 = self._data_1[0, :]
        if matched_only:
            for i in range(len(self._map)):
                if self._map[i, 0] == 0 or self._map[i, 1] == 0:
                    continue
                else:
                    dataset_0 = np.vstack((dataset_0, self._data_0[self._map[i, 0], :]))
                    dataset_1 = np.vstack((dataset_1, self._data_1[self._map[i, 1], :]))
        # Todo: need to note which cells  are matched in the matched_only == False option
        else:
            dataset_0_unmatched = self._data_0[0, :]
            dataset_1_unmatched = self._data_0[0, :]
            for i in range(len(self._map)):
                if self._map[i, 0] == 0:
                    dataset_1_unmatched = np.vstack((dataset_1_unmatched, self._data_1[self._map[i, 1], :]))
                elif self._map[i, 1] == 0:
                    dataset_0_unmatched = np.vstack((dataset_0_unmatched, self._data_0[self._map[i, 0], :]))
                else:
                    dataset_0 = np.vstack((dataset_0, self._data_0[self._map[i, 0], :]))
                    dataset_1 = np.vstack((dataset_1, self._data_1[self._map[i, 1], :]))
            dataset_0 = np.vstack((dataset_0, dataset_0_unmatched))
            dataset_1 = np.vstack((dataset_1, dataset_1_unmatched))
        setattr(self, f'_data_0', dataset_0)
        setattr(self, f'_data_1', dataset_1)
        data_list = [dataset_0, dataset_1]
        # Todo: update num_neurons so it is consistent with both datasets when all neurons are included
        self._num_neurons = np.shape(dataset_0)[0]
        # Add a series of private attributes which are CaGraph objects
        for i, dataset in enumerate(self._dataset_identifiers):
            setattr(self, f'__{dataset}_cagraph', CaGraph(data=data_list[i], correlation_method=correlation_method, threshold=self._threshold, **correlation_kwargs))

    # Private utility methods
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
    def dataset_identifiers(self):
        return self._dataset_identifiers

    def __input_validator(self, data, dataset_labels):
        """
            Performs input validation for CaGraphMatched class.

            :param data: list
                A list of dataset inputs, which can be either numpy.ndarray or file paths (str) to .csv or .nwb files.
            :param dataset_labels: list
                A list of labels for the datasets. The number of dataset labels must match the number of datasets in 'data'.

            :raises ValueError: If the number of dataset_labels does not match the number of datasets in 'data_list'.
            :raises TypeError: If data is not provided in the correct format, i.e., either as a str containing a .csv or .nwb file,
                or as numpy.ndarray.

            :return: None
            """
        if len(data) != len(dataset_labels):
            raise ValueError("The number of dataset_labels must match the number of datasets in 'data_list'")
        for i, dataset in enumerate(data):
            if isinstance(dataset, np.ndarray):
                setattr(self, f'_data_{i}', dataset)
            elif isinstance(dataset, str):
                if dataset.endswith('csv'):
                    setattr(self, f'_data_{i}', np.genfromtxt(dataset, delimiter=","))
                elif dataset.endswith('nwb'):
                    with NWBHDF5IO(dataset, 'r') as io:
                        nwbfile_read = io.read()
                        nwb_acquisition_key = list(nwbfile_read.acquisition.keys())[0]
                        ca_from_nwb = nwbfile_read.acquisition[nwb_acquisition_key]
                        setattr(self, f'_data_{i}', np.vstack((ca_from_nwb.timestamps[:], ca_from_nwb.data[:])))
                else:
                    raise TypeError('File path must have a .csv or .nwb file to load.')
            else:
                raise TypeError('Data must be passed as a str containing a .csv or .nwb file, or as numpy.ndarray.')

    def __generate_threshold(self, correlation_method='pearson', **correlation_kwargs) -> float:
        """
        Generates a threshold for the provided dataset as described in the preprocess module.
        This threshold generation will use the full dataset.

        :return: float
        """
        return preprocess.generate_average_threshold(data=self._data_0[1:, :], shuffle_iterations=10, correlation_method=correlation_method, **correlation_kwargs)

    # Public utility methods
    def save(self, file_path=None):
        """
        Save the CaGraphMatched object to a file.

        :param file_path: str, optional
            The path to the file where the object will be saved. If not provided, the default filename "obj.cagraph" is used.
        """
        if file_path is None:
            file_path = 'obj.cagraph'
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path):
        """
        Load a CaGraphMatched object from a file.

        :param file_path: str
            The path to the file from which to load the CaGraphMatched object.

        :return: CaGraphMatched
            The loaded CaGraphMatched object.
        """
        with open(file_path, 'rb') as file:
            cagraphtimesamples_obj = pickle.load(file)
        return cagraphtimesamples_obj

    def get_cagraph(self, condition_label):
        """
        Get the CaGraph object for the specified dataset condition.

        :param condition_label: str
            The label of the dataset condition for which to retrieve the CaGraph.

        :return: CaGraph
            The CaGraph object for the specified condition.
        """
        return getattr(self, f'__{condition_label}_cagraph')

    def get_full_report(self, analysis_selections=None, save_report=False, save_path=None, save_filename=None,
                        save_filetype=None):
        """
        Generates an organized report of all data in the batched sample and creates a comprehensive tabular summary.

        The function iterates through all datasets in the batched sample, retrieves reports from the respective
        CaGraph objects, and constructs a single pandas DataFrame that includes the reports for all datasets in a tabular structure.

        :param save_report: bool, optional
            If True, the generated report will be saved to a file.
        :param save_path: str, optional
            The directory path where the report file will be saved. If None, the current working directory is used.
        :param save_filename: str, optional
            The name of the saved report file (excluding file extension). If None, the default name 'report' is used.
        :param save_filetype: str, optional
            The file format for saving the report. Supported formats include 'csv', 'HDF5', and 'xlsx'. If None or not specified,
            the default format is 'csv'.

        :return: pandas.DataFrame
            A pandas DataFrame containing the combined reports for all datasets in a tabular format.

        :raises ValueError: If the save_filetype is provided but not supported (i.e., not 'csv', 'HDF5', or 'xlsx').

        Example:
        ```
        cagraph_batch = CaGraphBatch(data_list, dataset_labels)
        full_report = cagraph_batch.get_full_report(save_report=True, save_path='reports/', save_filename='full_report')
        ```
        """
        store_reports = {}
        for key in self._dataset_identifiers:
            cagraph_obj = self.get_cagraph(key)
            store_reports[key] = cagraph_obj.get_report()

        # For each column in the individual reports, append to the full report
        full_report_df = pd.DataFrame()
        for col in store_reports[key].columns:
            for key in store_reports.keys():
                df = store_reports[key]
                df = df.rename(columns={col: f'{key}_{col}'})
                full_report_df = pd.concat([full_report_df, df[f'{key}_{col}']], axis=1)

        if analysis_selections is not None:
            # Todo: Check that analysis_selections are valid
            selections_str = '|'.join(analysis_selections)
            full_report_df = full_report_df.filter(regex=selections_str)

        # Save the report
        if save_report:
            if save_filename is None:
                save_filename = 'report'
            if save_path is None:
                save_path = os.getcwd() + '/'
            elif not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            if save_filetype is None or save_filetype == 'csv':
                full_report_df.to_csv(save_path + save_filename + '.csv', index=True)
            elif save_filetype == 'HDF5':
                full_report_df.to_hdf(save_path + save_filename + '.h5', key=save_filename, mode='w')
            elif save_filetype == 'xlsx':
                full_report_df.to_excel(save_path + save_filename + 'xlsx', index=True)
        return full_report_df


# %% Behavior analysis
class CaGraphBehavior:
    """
    Class for running behavior-sampled analyses on a single dataset.

    This class is best suited for analyses where the time spent in each behavior is well-balanced, however, methods will
    be added to accommodate datasets with unbalanced behavior.
    """

    def __init__(self, data, behavior_data, behavior_dict, correlation_method='pearson', construction_method='stacked', node_labels=None,
                 node_metadata=None,
                 dataset_id=None, threshold=None,**correlation_kwargs):
        """
                Initialize a CaGraphBehavior object.

                :param data: str
                    The path to the dataset file.
                :param behavior_data: list
                    A list of behavior data that associates each time point with a behavior.
                :param behavior_dict: dict
                    A dictionary that maps behavior labels to behavior values.
                :param construction_method: str, optional
                    The method to construct CaGraph objects for different behaviors. Default is 'stacked'.
                :param node_labels: list, optional
                    A list of node labels for the CaGraph objects.
                :param node_metadata: dict, optional
                    A dictionary containing metadata for each node.
                :param dataset_id: str, optional
                    Identifier for the dataset.
                :param threshold: float, optional
                    The threshold to use for creating CaGraph objects. If not provided, it will be generated.

                :raises ValueError: If an invalid construction_method is provided.

                Example:
                ```
                behavior_data = [0, 0, 0, 1, 1, 1, 1, 0, 0, ..., 0, 0, 1, 1, 1, 1, 1]
                behavior_dict = {'freezing': 1, 'moving': 0}
                data_file = 'dataset.csv'
                cagraph_behavior = CaGraphBehavior(data_file, behavior_data, behavior_dict)
                ```
        """
        # Check that the input data is in the correct format and load dataset
        self.__input_validator(data=data)

        # Add dataset identifier
        if dataset_id is not None:
            self._data_id = dataset_id

        self._behavior_dict = behavior_dict
        self._behavior_data = np.loadtxt(behavior_data, delimiter=',')
        self._behavior_identifiers = list(behavior_dict.keys())

        # Compute time interval and number of neurons
        self._dt = self.data[0, 1] - self.data[0, 0]
        self._num_neurons = np.shape(self.data)[0]
        if threshold is not None:
            self._threshold = threshold
        else:
            self._threshold = self.__generate_threshold(correlation_method=correlation_method, **correlation_kwargs)

        # Generate node labels
        if node_labels is None:
            self._node_labels = np.linspace(0, np.shape(self.data)[0] - 2,
                                            np.shape(self.data)[0] - 1).astype(int)
        else:
            self._node_labels = node_labels

        # Add a series of private attributes which are CaGraph objects
        # Todo: change this method so it only appends when you switch from one behavior to the other (fewer append operations = faster)
        if construction_method == 'stacked':
            for key in behavior_dict.keys():
                # Build new behavior dataset
                behavior_dataset = np.ndarray((self._num_neurons, 1))
                behavior_value = behavior_dict[key]
                for i, value in enumerate(self._behavior_data):
                    if value == behavior_value:
                        # Append single timepoint
                        behavior_dataset = np.hstack((behavior_dataset, self._data[:, i].reshape(-1, 1)))
                setattr(self, f'__{key}_cagraph', CaGraph(data=behavior_dataset[:, 1:], node_labels=self._node_labels,
                                                          node_metadata=node_metadata, correlation_method=correlation_method, threshold=self._threshold, **correlation_kwargs))
        else:
            raise ValueError("Invalid value for construction_method. Must choose from: 'stacked'.")

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
    def behavior_identifiers(self):
        return self._behavior_identifiers

    def __input_validator(self, data):
        """
        Validate and load the input data for analysis.

        :param data: str or numpy.ndarray
            A string pointing to the file to be used for data analysis, or a numpy.ndarray containing data loaded into
            memory. The first (idx 0) row must contain timepoints, the subsequent rows each represent a single neuron
            timeseries of calcium fluorescence data sampled at the timepoints specified in the first row.
        :raises TypeError: If data is not a valid string path or a numpy.ndarray.
        """
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

    def __generate_threshold(self, correlation_method='pearson', **correlation_kwargs) -> float:
        """
        Generates a threshold for the provided dataset as described in the preprocess module.
        This threshold generation will use the full dataset.

        :return: float
        """
        return preprocess.generate_average_threshold(data=self.data[1:, :], shuffle_iterations=10, correlation_method=correlation_method, **correlation_kwargs)

    # Public utility methods
    def save(self, file_path=None):
        """
        Saves the CaGraphBehavior object to a binary file using pickle.

        :param file_path: str, optional
            The file path where the object will be saved. If not provided, the default filename "obj.cagraph" will be used.
        """
        if file_path is None:
            file_path = 'obj.cagraph'
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(file_path):
        """
        Loads a CaGraphBehavior object from a binary file using pickle.

        :param file_path: str
            The file path of the saved CaGraphBehavior object.

        :return: CaGraphBehavior
            The loaded CaGraphBehavior object.
        """
        with open(file_path, 'rb') as file:
            cagraphbehavior_obj = pickle.load(file)
        return cagraphbehavior_obj

    def get_cagraph(self, condition_label):
        """
        Retrieve a CaGraph object associated with a specific condition.

        :param condition_label: str
            The label of the condition for which to retrieve the CaGraph object.

        :return: CaGraph
            The CaGraph object associated with the specified condition.
        """
        return getattr(self, f'__{condition_label}_cagraph')

    def get_full_report(self, analysis_selections=None, save_report=False, save_path=None, save_filename=None,
                        save_filetype=None):
        """
            Generate an organized report of the CaGraph analyses for different behaviors.

            This method computes reports for each behavior condition and combines them into a single report, where each
            behavior condition's analyses are prefixed with the condition label in the column names.

            :param save_report: bool, optional
                If True, the report is saved to a file. Default is False.

            :param save_path: str, optional
                The path to save the report file. If None, the current working directory is used. Default is None.

            :param save_filename: str, optional
                The name of the report file. Default is 'report'.

            :param save_filetype: str, optional
                The file format to save the report. Supported types are 'csv', 'HDF5', and 'xlsx'. Default is 'csv'.

            :return: pandas.DataFrame
                A DataFrame containing the organized report of all data for different behaviors.
        """
        store_reports = {}
        for key in self._behavior_identifiers:
            cagraph_obj = self.get_cagraph(key)
            store_reports[key] = cagraph_obj.get_report()

        # For each column in the individual reports, append to the full report
        full_report_df = pd.DataFrame()
        for col in store_reports[key].columns:
            for key in store_reports.keys():
                df = store_reports[key]
                df = df.rename(columns={col: f'{key}_{col}'})
                full_report_df = pd.concat([full_report_df, df[f'{key}_{col}']], axis=1)

        if analysis_selections is not None:
            # Todo: Check that analysis_selections are valid
            selections_str = '|'.join(analysis_selections)
            full_report_df = full_report_df.filter(regex=selections_str)

        # Save the report
        if save_report:
            if save_filename is None:
                save_filename = 'report'
            if save_path is None:
                save_path = os.getcwd() + '/'
            elif not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            if save_filetype is None or save_filetype == 'csv':
                full_report_df.to_csv(save_path + save_filename + '.csv', index=True)
            elif save_filetype == 'HDF5':
                full_report_df.to_hdf(save_path + save_filename + '.h5', key=save_filename, mode='w')
            elif save_filetype == 'xlsx':
                full_report_df.to_excel(save_path + save_filename + 'xlsx', index=True)
        return full_report_df

# %% Remaining updates
# Todo: CaGraphBatch -> add option to add cell metadata
# Todo: CaGraphBatch -> ensure that datasets which are thrown out at initial CaGraph object generation are not included in future versions
# Todo: CaGraphBatch -> make a constructor function that can pass loaded data, numpy arrays
# Todo: All classes -> check docstrings again
# Todo: Add whole-graph analysis like report_dict['density']  = self.graph_theory.get_graph_density()
# Todo: CaGraphBatch -> high priority write second report method that averages results and stores the averages
# Todo: CaGraphTimeSamples -> Create a systematic return report/ dictionary
# Todo: Allow user to set multiple thresholds (< 0.1, > 0.5)
# Todo: CaGraphBehavior -> expand functionality
# Todo: CaGraphBatch derivatives: input should be able to be datasets = [np.ndarray or path...],  path], labels ['id', 'id']
