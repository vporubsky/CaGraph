"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 
File Final Edit Date:

Description: 
"""
from ca_graph import *

class DGGraph(CaGraph):
    """
    Class for LC-DG experiments. Context A and Context B are specified for
    fear conditioning paradigm, where Context A is anxiogenic and recorded from
    time 180 to 360 seconds and Context B is neutral and recorded from time
    0 to 180 seconds.

    Derived from parent class CaGraph
    """

    # Pass __init__ from parent class
    def __init__(self, data_file, identifiers=None, dataset_id=None, threshold=None):
        super().__init__(data_file, identifiers, dataset_id, threshold)
        self.context_A_dynamics = self.neuron_dynamics[:, 1800:3600]  # Record second in Context A
        self.context_B_dynamics = self.neuron_dynamics[:, 0:1800]  # Record first in Context B
        self.con_A_pearsons_correlation_matrix = np.corrcoef(self.context_A_dynamics)
        self.con_B_pearsons_correlation_matrix = np.corrcoef(self.context_B_dynamics)
    pass

    # Todo: Integrate cell-matched metadata -- consider adding this to ca_graph.py
    def get_cell_matched(self):
        """
        Returns a dictionary of neuron activity indices for day 1 and day 9
        for each mouse, for all neurons which remain in the field of view for
        both days.

        :return:
        """
        return

    def get_context_active(self, context_tag_data):
        """
        Returns lists indicating the indices of non-specific cells, context A-specific, or context B-specific cells.

        :return:
        """
        context_tags = list(np.genfromtxt(context_tag_data,delimiter=','))
        nonspecific_indices = [i for i, e in enumerate(context_tags) if e == 0]
        con_A_active_indices = [i for i, e in enumerate(context_tags) if e == 1]
        con_B_active_indices = [i for i, e in enumerate(context_tags) if e == 2]
        return nonspecific_indices, con_A_active_indices, con_B_active_indices

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
            plt.ylabel('ΔF/F')
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
            plt.ylabel('ΔF/F')
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

