"""
Perform PCA on a time-series dataset (calcium fluorescence data) to generate
a decomposition of the data to show the trajectory in 3D state space.


"""
from neuronal_network_graph import DGNetworkGraph as nng
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
import seaborn as sns
import matplotlib as mpl
import pandas as pd
import scipy
import random
from pynwb import NWBHDF5IO


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
    data_file : str
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

    # Todo: tell what U, Sig and V are
    def get_SVD(self):
        """
        Returns the singular value decomposition of the neuronal dynamics as U, Sig, and V matrices.

        :return: U, Sig, and V matrices
        """
        neuron_dynamics = np.transpose(self.neuron_dynamics)
        neuron_dynamics -= np.mean(neuron_dynamics)
        neuron_dynamics /= np.std(neuron_dynamics)

        # SVD returns U, Sig, and V matrices
        return svd(neuron_dynamics)

    def project_to_subspace(self, dimensions=3):
        """
        Project dataset to "n" reduced dimensions.

        :param dimensions: int
        :return:
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

    def plot_3D_projection(self, linestyle='.', color='k', alpha=0.5):
        """
        Plot 3-dimensional projection of neuronal dynamics.

        :param linestyle: matplotlib linestyle options
        :param color: matplotlib color options
        :param alpha: matplotlib alpha transparency options
        """
        projection = self.project_to_subspace(dimensions=3)
        ax = plt.axes(projection='3d')
        ax.scatter3D(projection[:, 0], projection[:, 1], projection[:, 2], linestyle, c=color, alpha=alpha)


# %% Wildtype condition
D1_WT = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv',
         '1055-4_D1_smoothed_calcium_traces.csv',
         '122-1_D1_smoothed_calcium_traces.csv',
         '122-2_D1_smoothed_calcium_traces.csv', '122-3_D1_smoothed_calcium_traces.csv',
         '14-0_D1_smoothed_calcium_traces.csv']
D9_WT = ['1055-1_D9_smoothed_calcium_traces.csv', '1055-2_D9_smoothed_calcium_traces.csv',
         '1055-4_D9_smoothed_calcium_traces.csv',
         '14-0_D9_smoothed_calcium_traces.csv', '122-1_D9_smoothed_calcium_traces.csv',
         '122-2_D9_smoothed_calcium_traces.csv', '122-3_D9_smoothed_calcium_traces.csv']

path_to_data = "/LC-DG-FC-data/"

data_file = D1_WT[3]
mouse_id = data_file.replace('_smoothed_calcium_traces.csv', '')
nn = nng(path_to_data + data_file)

# %% Normalize Data
# Todo: normalize as necessary

# %% Compute covariance matrix
data = nn.neuron_dynamics
cov = (1 / len(data)) * np.dot(data, np.transpose(data))
plt.imshow(cov, cmap='hot')
plt.show()

# %% Eigenvalue decomposition
# data = np.transpose(nn.neuron_dynamics)
# C = np.matmul(data, np.transpose(data))
# Eig_Val, Eig_Vec = np.linalg.eig(C)
#
# # Project to 3D
# V_r = Eig_Vec[:, 0:3]
# projection = np.matmul(nn.neuron_dynamics, V_r)
#
# # Plot projection/ trajectory in state space
# ax = plt.axes(projection='3d')
# ax.scatter3D(projection[:,0], projection[:,1], projection[:,2], '.');
# plt.show()

# %% Singular value decomposition in two dimensions
# Todo: check normalization
mouse_idx = 2
day = 'D1'
if day == 'D1':
    data_file = D1_WT[mouse_idx]
elif day == 'D9':
    data_file = D9_WT[mouse_idx]

mouse_id = data_file.replace('_smoothed_calcium_traces.csv', '')
nn = nng(path_to_data + data_file)

data_A = np.transpose(nn.context_A_dynamics)
data_A -= np.mean(data_A)
data_A /= np.std(data_A)
data_B = np.transpose(nn.context_B_dynamics)
data_B -= np.mean(data_B)
data_B /= np.std(data_B)

# SVD
U_a, Sig_a, V_a = svd(data_A)
U_b, Sig_b, V_b = svd(data_B)

# Project dataset to 2 dimensions
V_r_a = V_a[:, 0:2]
V_r_b = V_b[:, 0:2]
projection_a = np.matmul(data_A, V_r_a)
projection_b = np.matmul(data_B, V_r_b)

# Plot projection/ trajectory in state space
plt.title(f"{mouse_id} on {day}")
plt.plot(projection_b[:, 0], projection_b[:, 1], '-', c='teal', alpha=0.5);
plt.plot(projection_a[:, 0], projection_a[:, 1], '-', c='salmon', alpha=0.5);
plt.show()

# %% test function definition
test_class = NeuronalDynamicsProjection(data=nn.data)
U, Sig, V = test_class.get_SVD()
test_class.plot_2D_projection(alpha=0.1)
plt.show()

# %% SVD in three dimensions
V_r_a = V_a[:, 0:3]
V_r_b = V_b[:, 0:3]
projection_a = np.matmul(data_A, V_r_a)
projection_b = np.matmul(data_B, V_r_b)
# Plot projection/ trajectory in state space
ax = plt.axes(projection='3d')
ax.scatter3D(projection_a[:, 0], projection_a[:, 1], projection_a[:, 2], '.', c='salmon');
ax.scatter3D(projection_b[:, 0], projection_b[:, 1], projection_b[:, 2], '.', c='teal');
plt.show()

# %% Singular value decomposition in two dimensions
# Todo: check normalization
ax = plt.axes(projection='3d')
mouse_idx = 6
data_file = D1_WT[mouse_idx]

mouse_id = data_file.replace('_smoothed_calcium_traces.csv', '')
nn = nng(path_to_data + data_file)

data_A = np.transpose(nn.context_A_dynamics)
data_A -= np.mean(data_A)
data_A /= np.std(data_A)
data_B = np.transpose(nn.context_B_dynamics)
data_B -= np.mean(data_B)
data_B /= np.std(data_B)

# SVD
U_a, Sig_a, V_a = svd(data_A)
U_b, Sig_b, V_b = svd(data_B)

V_r_a = V_a[:, 0:3]
V_r_b = V_b[:, 0:3]
projection_a1 = np.matmul(data_A, V_r_a)
projection_b1 = np.matmul(data_B, V_r_b)

data_file = D9_WT[mouse_idx]
mouse_id = data_file.replace('_smoothed_calcium_traces.csv', '')
nn = nng(path_to_data + data_file)

data_A = np.transpose(nn.context_A_dynamics)
data_A -= np.mean(data_A)
data_A /= np.std(data_A)
data_B = np.transpose(nn.context_B_dynamics)
data_B -= np.mean(data_B)
data_B /= np.std(data_B)

# SVD
U_a, Sig_a, V_a = svd(data_A)
U_b, Sig_b, V_b = svd(data_B)

V_r_a = V_a[:, 0:3]
V_r_b = V_b[:, 0:3]
projection_a9 = np.matmul(data_A, V_r_a)
projection_b9 = np.matmul(data_B, V_r_b)

# Plot projection/ trajectory in state space

ax.scatter3D(projection_a1[:, 0], projection_a1[:, 1], projection_a1[:, 2], '.', color='salmon');
ax.scatter3D(projection_b1[:, 0], projection_b1[:, 1], projection_b1[:, 2], '.', color='teal');

ax.scatter3D(projection_a9[:, 0], projection_a9[:, 1], projection_a9[:, 2], '.', color='mistyrose');
ax.scatter3D(projection_b9[:, 0], projection_b9[:, 1], projection_b9[:, 2], '.', color='turquoise');
plt.show()

# %% Singular value decomposition in two dimensions with plotly express
import plotly.express as px

ax = plt.axes(projection='3d')
mouse_idx = 6
data_file = D1_WT[mouse_idx]

mouse_id = data_file.replace('_smoothed_calcium_traces.csv', '')
nn = nng(path_to_data + data_file)

data_A = np.transpose(nn.context_A_dynamics)
data_A -= np.mean(data_A)
data_A /= np.std(data_A)
data_B = np.transpose(nn.context_B_dynamics)
data_B -= np.mean(data_B)
data_B /= np.std(data_B)

# SVD
U_a, Sig_a, V_a = svd(data_A)
U_b, Sig_b, V_b = svd(data_B)

V_r_a = V_a[:, 0:3]
V_r_b = V_b[:, 0:3]
projection_a1 = np.matmul(data_A, V_r_a)
projection_b1 = np.matmul(data_B, V_r_b)

data_file = D9_WT[mouse_idx]
mouse_id = data_file.replace('_smoothed_calcium_traces.csv', '')
nn = nng(path_to_data + data_file)

data_A = np.transpose(nn.context_A_dynamics)
data_A -= np.mean(data_A)
data_A /= np.std(data_A)
data_B = np.transpose(nn.context_B_dynamics)
data_B -= np.mean(data_B)
data_B /= np.std(data_B)

# SVD
U_a, Sig_a, V_a = svd(data_A)
U_b, Sig_b, V_b = svd(data_B)

V_r_a = V_a[:, 0:3]
V_r_b = V_b[:, 0:3]
projection_a9 = np.matmul(data_A, V_r_a)
projection_b9 = np.matmul(data_B, V_r_b)

data_a1 = [[projection_a1[:, 0], projection_a1[:, 1], projection_a1[:, 2]]]
df_a1 = pd.DataFrame(data_a1, columns=['PC1', 'PC2', 'PC3'])
print(df_a1)

fig = px.scatter_3d(df_a1, x='PC1', y='PC2', z='PC3', color='PC1', symbol='PC1')
fig.show
fig.show()
# Plot projection/ trajectory in state space

ax.scatter3D(projection_a1[:, 0], projection_a1[:, 1], projection_a1[:, 2], '.', color='salmon');
ax.scatter3D(projection_b1[:, 0], projection_b1[:, 1], projection_b1[:, 2], '.', color='teal');

ax.scatter3D(projection_a9[:, 0], projection_a9[:, 1], projection_a9[:, 2], '.', color='mistyrose');
ax.scatter3D(projection_b9[:, 0], projection_b9[:, 1], projection_b9[:, 2], '.', color='turquoise');
plt.show()

# %% Look at only cells that are preferentially active in a given context
