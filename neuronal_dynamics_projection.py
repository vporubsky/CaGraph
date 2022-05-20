"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 2022-04-25
File Final Edit Date:

Description: A class to perform dimensionality reduction. Currently only implementing PCA.
"""
from pynwb import NWBHDF5IO
import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.ndimage.filters import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")


class NeuronalDynamicsProjection:
    """
    Ongoing development: 02/30/2022
    Published: XX/XX/XXXX
    Author: Veronica Porubsky [Github: https://github.com/vporubsky][ORCID: https://orcid.org/0000-0001-7216-3368]

    Class: NeuronalDynamicsProjection()
    =====================

    This class provides functionality to reduce the dimensionality
    of neuronal dynamics from calcium imaging data.

    Attributes
    ----------
    data : str
        A string pointing to the file to be used for data analysis.

    Methods
    """

    def __init__(self, data):
        """
        :param data:
        """
        if type(data) == str and data.endswith('csv'):
            self.data = np.genfromtxt(data, delimiter=",")
            self.time = self.data[0, :]
            self.neuron_dynamics = self.data[1:len(self.data), :]
        elif type(data) == str and data.endswith('nwb'):
            with NWBHDF5IO(data, 'r') as io:
                nwbfile_read = io.read()
                nwb_acquisition_key = list(nwbfile_read.acquisition.keys())[0]
                ca_from_nwb = nwbfile_read.acquisition[nwb_acquisition_key]
                self.data = data
                self.neuron_dynamics = ca_from_nwb.data[:]
                self.time = ca_from_nwb.timestamps[:]
        elif type(data) == np.ndarray:
            self.data = data
            self.neuron_dynamics = data[1:, :]
            self.time = data[0, :]
        else:
            print('Data must be passed as a .csv or .nwb file, or as a numpy.ndarray containing the neuronal dynamics'
                  'as individual rows in the array with the time data in the row at index 0.')
            raise TypeError
        self.num_neurons = np.shape(self.neuron_dynamics)[0]

    def get_SVD(self):
        """
        Returns the singular value decomposition of the neuronal dynamics as U, Sig, and V matrices.
        The singular value decomposition is a generalization of the eigendecomposition of a square normal matrix with
        an orthonormal eigenbasis -- it can accomodate any mxn matrix.

        U: an mxm complex unitary matrix (eigen neurons -- tells you about the column space of the data)
        Sig: an mxn rectangular diagonal matrix with non-negative real numbers along the diagonal
        V: an nxn complex unitary matrix (eigen timeseries -- tells you about the row space of the data )

        :return: U, Sig, and V matrices
        """
        neuron_dynamics = np.transpose(self.neuron_dynamics)
        neuron_dynamics -= np.mean(neuron_dynamics)
        neuron_dynamics /= np.std(neuron_dynamics)

        # SVD returns U, Sig, and V matrices
        return svd(neuron_dynamics)

    def neural_svd(self, dimensions=3):
        """
        Project dataset to "n" reduced dimensions using the singular value decomposition. Returns projected data.

        :param dimensions: int
        :return: np.ndarray
        """
        neuron_dynamics = np.transpose(self.neuron_dynamics)
        neuron_dynamics -= np.mean(neuron_dynamics)
        neuron_dynamics /= np.std(neuron_dynamics)
        U, Sig, V = self.get_SVD()
        # Reduced subset of V
        V_r = V[:, 0:dimensions]
        return np.matmul(neuron_dynamics, V_r)

    def plot_2D_projection(self, linestyle='.', color='k', alpha=0.5):
        """
        Plot 2-dimensional projection of neuronal dynamics.

        :param linestyle: matplotlib linestyle options
        :param color: matplotlib color options
        :param alpha: matplotlib alpha transparency options
        """
        projection = self.project_to_subspace(dimensions=2)
        plt.plot(projection[:, 0], projection[:, 1], linestyle, c=color, alpha=alpha);

    def plot_3D_projection_deprecated(self, linestyle='.', color='k', alpha=0.5):
        """
        Plot 3-dimensional projection of neuronal dynamics.

        :param linestyle: matplotlib linestyle options
        :param color: matplotlib color options
        :param alpha: matplotlib alpha transparency options
        """
        projection = self.project_to_subspace(dimensions=3)
        ax = plt.axes(projection='3d')
        ax.scatter3D(projection[:, 0], projection[:, 1], projection[:, 2], linestyle, c=color, alpha=alpha)

    def z_score(self, dataset=None):
        """
        Computes the z-score (zero mean and unit variance) of the dataset for normalization
        and centering.
        """
        if dataset is None:
            data = self.neuron_dynamics
        else:
            data = dataset
        ss = StandardScaler(with_mean=True, with_std=True)
        z_scored_dynamics = ss.fit_transform(data.T).T
        return z_scored_dynamics

    def neural_pca(self, dataset=None, num_components=3):
        """
        Computes the PCA of the dataset.

        :param dataset: np.ndarray
        :param num_components: int
        """
        # standardize data with z-score and apply PCA
        if dataset is not None:
            z_scored_dynamics = self.z_score(dataset)
        else:
            z_scored_dynamics = self.z_score(self.neuron_dynamics)

        pca = PCA(n_components=num_components)
        pca_dynamics = pca.fit_transform(z_scored_dynamics.T).T
        return pca_dynamics

    # utility function
    def __3d_ax_stylesheet(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        ax.set_xlabel('PC 1')
        ax.set_ylabel('PC 2')
        ax.set_zlabel('PC 3')

    def plot_3D_projection(self, dataset=None, num_components=3, axs=None, color='k', return_axes=True):
        """
        Plot 3-dimensional projection of neuronal dynamics.
        """
        sigma = 3  # smoothing amount
        # specify component indices
        component_x = 0
        component_y = 1
        component_z = 2

        # set up a figure with two 3d subplots with distinct views if axes not supplied
        if axs is not None:
            ax1, ax2 = axs
        else:
            fig= plt.figure(figsize=[9, 4])
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            axs = [ax1, ax2]

        pca_dynamics = self.neural_pca(dataset=dataset, num_components=num_components)

        for ax in axs:
            x = pca_dynamics[component_x, :]
            y = pca_dynamics[component_y, :]
            z = pca_dynamics[component_z, :]

            # apply Gaussian smoothing to the trajectories
            x = gaussian_filter1d(x, sigma=sigma)
            y = gaussian_filter1d(y, sigma=sigma)
            z = gaussian_filter1d(z, sigma=sigma)

            ax.plot(x, y, z, c=color)

            # plot circles at beginning of trajectories
            ax.scatter(x[0], y[0], z[0], c=color, s=14)

            # make the axes cleaner
            self.__3d_ax_stylesheet(ax)

        # specify the orientation of the 3d plot
        ax1.view_init(elev=22, azim=30)
        ax2.view_init(elev=22, azim=110)

        plt.tight_layout()

        if return_axes:
            return [ax1, ax2]

