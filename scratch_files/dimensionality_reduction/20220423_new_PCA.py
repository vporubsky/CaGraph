"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 
File Final Edit Date:

Description: An additional test of PCA projection.

Based on tutorial by:https://pietromarchesi.net/pca-neural-data.html

"""
import numpy as np
from sklearn.preprocessing import StandardScaler

def center(X):
    # X: ndarray, shape (n_features, n_samples)
    ss = StandardScaler(with_mean=True, with_std=False)
    Xc = ss.fit_transform(X.T).T
    return Xc

X = np.array([[5., 6., 5.],
              [50., 60., 50.]]) # data array of shape n_neuron by n_time_points
X = center(X)
c = np.cov(X, rowvar=True) # covariance matrix
eig_vals, eig_vecs = np.linalg.eig(c)
srt = np.argsort(eig_vals)[::-1]
print('Sorted eigenvalues: \n{}'.format(eig_vals[srt]))
print('\nFirst eigenvector: \n{}'.format(eig_vecs[:, srt][:, 0]))


#%%
def z_score(X):
    # X: ndarray, shape (n_features, n_samples)
    ss = StandardScaler(with_mean=True, with_std=True)
    Xz = ss.fit_transform(X.T).T
    return Xz

X = np.array([[5., 6., 5.],
              [50., 60., 50.]]) # data array of shape n_neuron by n_time_points
X = z_score(X)
c = np.cov(X, rowvar=True) # covariance matrix
eig_vals, eig_vecs = np.linalg.eig(c)
srt = np.argsort(eig_vals)[::-1]
print('Sorted eigenvalues: \n{}'.format(eig_vals[srt]))
print('\nFirst eigenvector: \n{}'.format(eig_vecs[:, srt][:, 0]))

#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.ndimage.filters import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D


#%% figure out how to put my data here
from ca_graph import DGNetworkGraph as nng

frames_pre_stim = 6000
frames_post_stim = 6000

nn = nng('/Users/veronica_porubsky/GitHub/BLA_graph_theory/EZM/data/119-0_deconTrace.csv')

session_length = nn.time[-1]
rec_freq     = 1/(nn.time[1] - nn.time[0]) # 10 Hz
trial_type   = 'EZM'
trial_types  = ['EZM']
trials       = nn.neuron_dynamics
start_stim   = 600 # index at which stimulation is applied
end_stim     = 1200 # index at which stimulation is removed
session_type = '1p imaging'
time         = nn.time[-1]
trial_size   = 1
Nneurons     = nn.num_neurons


#%%
shade_alpha = 0.2
lines_alpha = 0.8
pal = sns.color_palette('husl', 9)


def add_stim_to_plot(ax):
    ax.axvspan(start_stim, end_stim, alpha=shade_alpha,
               color='gray')
    ax.axvline(start_stim, alpha=lines_alpha, color='gray', ls='--')
    ax.axvline(end_stim, alpha=lines_alpha, color='gray', ls='--')


def add_orientation_legend(ax):
    custom_lines = [Line2D([0], [0], color=pal[k], lw=4) for
                    k in range(len(trial_types))]
    labels = ['{}'.format(t) for t in trial_types]
    ax.legend(custom_lines, labels,
              frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout(rect=[0, 0, 0.9, 1])


#%% Trial-response PCA
# [:, frames_pre_stim:-frames_post_stim]
Xr = np.vstack([trials.mean(axis=1)]).T

# or take the max
#Xr = np.vstack([t[:, frames_pre_stim:-frames_post_stim].max(axis=1) for t in trials]).T

# or the baseline-corrected mean
#Xr = np.vstack([t[:, frames_pre_stim:-frames_post_stim].mean(axis=1) - t[:, 0:frames_pre_stim].mean(axis=1) for t in trials]).T

Xr_sc = z_score(Xr)

pca = PCA(n_components=1)
Xp = pca.fit_transform(Xr_sc.T).T

projections = [(0, 1), (1, 2), (0, 2)]
fig, axes = plt.subplots(1, 3, figsize=[9, 3], sharey='row', sharex='row')
for ax, proj in zip(axes, projections):
    x = Xp[proj[0]]
    y = Xp[proj[1]]
    ax.scatter(x, y, c=pal[0], s=25, alpha=0.8)
    ax.set_xlabel('PC {}'.format(proj[0]+1))
    ax.set_ylabel('PC {}'.format(proj[1]+1))
sns.despine(fig=fig, top=True, right=True)
add_orientation_legend(axes[2])

#%%
n_components = 15
Xa = z_score(trials) #Xav_sc = center(Xav)
pca = PCA(n_components=n_components)
Xa_p = pca.fit_transform(Xa.T).T

fig, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey='row')
for comp in range(3):
    ax = axes[comp]
    for kk, type in enumerate(trial_types):
        x = Xa_p[comp, :]
        x = gaussian_filter1d(x, sigma=3)
        ax.plot(nn.time, x, c=pal[kk])
    add_stim_to_plot(ax)
    ax.set_ylabel('PC {}'.format(comp+1))
add_orientation_legend(axes[2])
axes[1].set_xlabel('Time (s)')
sns.despine(fig=fig, right=True, top=True)
plt.tight_layout(rect=[0, 0, 0.9, 1])


#%% Loop through each mouse
import os
frames_pre_stim = 6000
frames_post_stim = 6000

nn = nng('/Users/veronica_porubsky/GitHub/BLA_graph_theory/EZM/data/119-0_deconTrace.csv')

session_length = nn.time[-1]
rec_freq     = 1/(nn.time[1] - nn.time[0]) # 10 Hz
trial_type   = 'EZM'
trial_types  = ['EZM']
trials       = nn.neuron_dynamics
start_stim   = 600 # index at which stimulation is applied
end_stim     = 1200 # index at which stimulation is removed
session_type = '1p imaging'
time         = nn.time[-1]
trial_size   = 1
Nneurons     = nn.num_neurons


datasets = os.listdir('/Users/veronica_porubsky/GitHub/BLA_graph_theory/EZM/data')
data_list = []
for dataset in datasets:
    if dataset.endswith('_deconTrace.csv'):
        data_list.append(dataset)

n_components = 15
Xa = z_score(trials) #Xav_sc = center(Xav)
pca = PCA(n_components=n_components)
Xa_p = pca.fit_transform(Xa.T).T

fig, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey='row')
for comp in range(3):
    ax = axes[comp]
    for kk, dataset in enumerate(data_list):
        nn = nng('/Users/veronica_porubsky/GitHub/BLA_graph_theory/EZM/data/' + dataset)
        Xa = z_score(nn.neuron_dynamics)  # Xav_sc = center(Xav)
        pca = PCA(n_components=n_components)
        Xa_p = pca.fit_transform(Xa.T).T
        x = Xa_p[comp, :]
        x = gaussian_filter1d(x, sigma=6)
        ax.plot(nn.time, x, c=pal[kk], alpha=0.4)
    add_stim_to_plot(ax)
    ax.set_ylabel('PC {}'.format(comp+1))
add_orientation_legend(axes[2])
axes[1].set_xlabel('Time (s)')
sns.despine(fig=fig, right=True, top=True)
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()


#%% Loop through each mouse for LC-DG-FC data

frames_pre_stim = 6000
frames_post_stim = 6000

nn = nng('/Users/veronica_porubsky/GitHub/BLA_graph_theory/EZM/data/119-0_deconTrace.csv')

session_length = nn.time[-1]
rec_freq     = 1/(nn.time[1] - nn.time[0]) # 10 Hz
trial_type   = 'EZM'
trial_types  = ['EZM']
trials       = nn.neuron_dynamics
start_stim   = 0 # index at which stimulation is applied
end_stim     = 0 # index at which stimulation is removed
session_type = '1p imaging'
time         = nn.time[-1]
trial_size   = 1
Nneurons     = nn.num_neurons


data_list = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv',
         '1055-4_D1_smoothed_calcium_traces.csv',
         '122-1_D1_smoothed_calcium_traces.csv',
         '122-2_D1_smoothed_calcium_traces.csv', '122-3_D1_smoothed_calcium_traces.csv',
         '14-0_D1_smoothed_calcium_traces.csv']

trial_averages = []
for ind, dataset in enumerate(data_list):
    nn = nng(os.getcwd() + '/LC-DG-FC-data/' + dataset)
    trial_averages.append(np.array(nn.context_A_dynamics[:, 0:1790]).mean(axis=0))
Xa = np.hstack(trial_averages)

n_components = 15
Xa = z_score(Xa)
pca = PCA(n_components=n_components)
Xa_p = pca.fit_transform(Xa.T).T

fig, axes = plt.subplots(1, 3, figsize=[10, 2.8], sharey='row')
for comp in range(3):
    ax = axes[comp]
    for kk, dataset in enumerate(data_list):

        Xa = z_score(nn.context_A_dynamics)  # Xav_sc = center(Xav)
        pca = PCA(n_components=n_components)
        Xa_p = pca.fit_transform(Xa.T).T
        x = Xa_p[comp, :]
        x = gaussian_filter1d(x, sigma=6)
        ax.plot(nn.time[0:len(x)], x, c=pal[kk], alpha=0.4)
    add_stim_to_plot(ax)
    ax.set_ylabel('PC {}'.format(comp+1))
add_orientation_legend(axes[2])
axes[1].set_xlabel('Time (s)')
sns.despine(fig=fig, right=True, top=True)
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()



#%% Trial-concatenated


data_list = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv',
         '1055-4_D1_smoothed_calcium_traces.csv',
         '122-1_D1_smoothed_calcium_traces.csv',
         '122-2_D1_smoothed_calcium_traces.csv', '122-3_D1_smoothed_calcium_traces.csv',
         '14-0_D1_smoothed_calcium_traces.csv']

trials = []
trial_types = []
trial_size = []
for dataset in data_list:
    nn = nng(os.getcwd() + '/LC-DG-FC-data/' + dataset)
    trials.append(np.array(nn.context_A_dynamics[:, 0:1790]).T)
    trial_types.append(dataset.replace('_D1_smoothed_calcium_traces.csv', ''))
    trial_size.append(nn.num_neurons)
Xl = np.hstack(trials)
Xl = Xl.T

Xl = z_score(Xl)
pca = PCA(n_components=15)
Xl_p = pca.fit_transform(Xl.T).T

gt = {comp : {t_type : [] for t_type in trial_types} for comp in range(n_components)}

for comp in range(n_components):
    for i, t_type in enumerate(trial_types):
        t = Xl_p[comp, :]
        gt[comp][t_type].append(t)

for comp in range(n_components):
    for t_type in trial_types:
        gt[comp][t_type] = np.vstack(gt[comp][t_type])

#%%
data_list = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv',
         '1055-4_D1_smoothed_calcium_traces.csv', '122-1_D1_smoothed_calcium_traces.csv',
         '122-2_D1_smoothed_calcium_traces.csv', '122-3_D1_smoothed_calcium_traces.csv',
         '14-0_D1_smoothed_calcium_traces.csv']

trials = []
trial_types = []
trial_size = []
f, axes = plt.subplots(1, 3, figsize=[10, 5], sharey=True, sharex=True)
for t, dataset in enumerate(data_list):
    nn = nng(os.getcwd() + '/LC-DG-FC-data/' + dataset)
    trial_types.append(dataset.replace('_D1_smoothed_calcium_traces.csv', ''))
    trial_size.append(nn.num_neurons)
    Xl = z_score(nn.neuron_dynamics[:, 0:3590])
    pca = PCA(n_components=15)
    Xl_p = pca.fit_transform(Xl.T).T
    for comp in range(3):
        ax = axes[comp]
        ax.plot(nn.time[0:3590], Xl_p[comp, :], color=pal[t])
        ax.set_ylabel('PC {}'.format(comp+1))

axes[1].set_xlabel('Time (s)')
sns.despine(right=True, top=True)
add_orientation_legend(axes[2])
plt.show()



#%% 3D plotting

fig = plt.figure(figsize=[9, 4])
trials = []
trial_types = []
trial_size = []
for dataset in data_list:
    nn = nng(os.getcwd() + '/LC-DG-FC-data/' + dataset)
    trials.append(np.array(nn.context_A_dynamics[:, 0:1790]).mean(axis=0))
    trial_types.append(dataset.replace('_D1_smoothed_calcium_traces.csv', ''))
    trial_size.append(nn.num_neurons)

Xa = trials[0]
for idx, trial in enumerate(trials):
    if idx != 0:
        Xa = np.vstack((Xa, trial))

# standardize and apply PCA
Xa = z_score(Xa)
pca = PCA(n_components=3)
Xa_p = pca.fit_transform(Xa.T).T

# pick the components corresponding to the x, y, and z axes
component_x = 0
component_y = 1
component_z = 2


# utility function to clean up and label the axes
def style_3d_ax(ax):
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


sigma = 3  # smoothing amount

# set up a figure with two 3d subplots, so we can have two different views

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
axs = [ax1, ax2]

for ax in axs:
    for t, t_type in enumerate(trial_types):
        # for every trial type, select the part of the component
        # which corresponds to that trial type:
        x = Xa_p[component_x, :]
        y = Xa_p[component_y, :]
        z = Xa_p[component_z, :]

        # apply some smoothing to the trajectories
        x = gaussian_filter1d(x, sigma=sigma)
        y = gaussian_filter1d(y, sigma=sigma)
        z = gaussian_filter1d(z, sigma=sigma)

        # use the mask to plot stimulus and pre/post stimulus separately
        # z_stim = z.copy()
        # z_stim[stim_mask] = np.nan
        # z_prepost = z.copy()
        # z_prepost[~stim_mask] = np.nan

        ax.plot(x, y, z, c=pal[t])

        # plot dots at initial point
        ax.scatter(x[0], y[0], z[0], c=pal[t], s=14)

        # make the axes a bit cleaner
        style_3d_ax(ax)

# specify the orientation of the 3d plot
ax1.view_init(elev=22, azim=30)
ax2.view_init(elev=22, azim=110)

trials = []
trial_types = []
trial_size = []
for dataset in data_list:
    nn = nng(os.getcwd() + '/LC-DG-FC-data/' + dataset)
    trials.append(np.array(nn.context_B_dynamics[:, 0:1790]).mean(axis=0))
    trial_types.append(dataset.replace('_D1_smoothed_calcium_traces.csv', ''))
    trial_size.append(nn.num_neurons)

Xa = trials[0]
for idx, trial in enumerate(trials):
    if idx != 0:
        Xa = np.vstack((Xa, trial))

# standardize and apply PCA
Xa = z_score(Xa)
pca = PCA(n_components=3)
Xa_p = pca.fit_transform(Xa.T).T

# pick the components corresponding to the x, y, and z axes
component_x = 0
component_y = 1
component_z = 2


for ax in axs:
    for t, t_type in enumerate(trial_types):
        # for every trial type, select the part of the component
        # which corresponds to that trial type:
        x = Xa_p[component_x, :]
        y = Xa_p[component_y, :]
        z = Xa_p[component_z, :]

        # apply some smoothing to the trajectories
        x = gaussian_filter1d(x, sigma=sigma)
        y = gaussian_filter1d(y, sigma=sigma)
        z = gaussian_filter1d(z, sigma=sigma)

        ax.plot(x, y, z, c=pal[2])

        # plot dots at initial point
        ax.scatter(x[0], y[0], z[0], c=pal[2], s=14)

        # make the axes a bit cleaner
        style_3d_ax(ax)

# specify the orientation of the 3d plot
ax1.view_init(elev=22, azim=30)
ax2.view_init(elev=22, azim=110)

plt.tight_layout()


#%% Finalize plotting
day = 'D9'
if day == 'D1':
    data_list = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv',
             '1055-4_D1_smoothed_calcium_traces.csv', '122-1_D1_smoothed_calcium_traces.csv',
             '122-2_D1_smoothed_calcium_traces.csv', '122-3_D1_smoothed_calcium_traces.csv',
             '14-0_D1_smoothed_calcium_traces.csv']
elif day == 'D9':
    data_list = ['1055-1_D9_smoothed_calcium_traces.csv', '1055-2_D9_smoothed_calcium_traces.csv',
             '1055-4_D9_smoothed_calcium_traces.csv', '122-1_D9_smoothed_calcium_traces.csv',
             '122-2_D9_smoothed_calcium_traces.csv', '122-3_D9_smoothed_calcium_traces.csv',
             '14-0_D9_smoothed_calcium_traces.csv']
for dataset in data_list:
    fig = plt.figure(figsize=[9, 4])
    nn = nng(os.getcwd() + '/LC-DG-FC-data/' + dataset)
    mouse_id = dataset.replace(f"_{day}_smoothed_calcium_traces.csv", "")
    Xa = nn.context_A_dynamics[:, 0:1790]
    # standardize and apply PCA
    Xa = z_score(Xa)
    pca = PCA(n_components=3)
    Xa_p = pca.fit_transform(Xa.T).T

    # Calculate the variance explained by priciple components
    print('Variance of each component:', pca.explained_variance_ratio_)
    print('\n Total Variance Explained:', round(sum(list(pca.explained_variance_ratio_)) * 100, 2))

    # pick the components corresponding to the x, y, and z axes
    component_x = 0
    component_y = 1
    component_z = 2


    # utility function to clean up and label the axes
    def style_3d_ax(ax):
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


    sigma = 3  # smoothing amount

    # set up a figure with two 3d subplots, so we can have two different views

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    axs = [ax1, ax2]

    for ax in axs:
        for t, t_type in enumerate(trial_types):
            # for every trial type, select the part of the component
            # which corresponds to that trial type:
            x = Xa_p[component_x, :]
            y = Xa_p[component_y, :]
            z = Xa_p[component_z, :]

            # apply some smoothing to the trajectories
            x = gaussian_filter1d(x, sigma=sigma)
            y = gaussian_filter1d(y, sigma=sigma)
            z = gaussian_filter1d(z, sigma=sigma)


            ax.plot(x, y, z, c='salmon')

            # plot dots at initial point
            ax.scatter(x[0], y[0], z[0], c='salmon', s=14)

            # make the axes a bit cleaner
            style_3d_ax(ax)

    # specify the orientation of the 3d plot
    ax1.view_init(elev=22, azim=30)
    ax2.view_init(elev=22, azim=110)

    # Load dataset
    Xa = nn.context_B_dynamics[:, 0:1790]
    # standardize and apply PCA
    Xa = z_score(Xa)
    pca = PCA(n_components=3)
    Xa_p = pca.fit_transform(Xa.T).T

    # Calculate the variance explained by priciple components
    print('Variance of each component:', pca.explained_variance_ratio_)
    print('\n Total Variance Explained:', round(sum(list(pca.explained_variance_ratio_)) * 100, 2))

    # pick the components corresponding to the x, y, and z axes
    component_x = 0
    component_y = 1
    component_z = 2


    for ax in axs:
        for t, t_type in enumerate(trial_types):
            # for every trial type, select the part of the component
            # which corresponds to that trial type:
            x = Xa_p[component_x, :]
            y = Xa_p[component_y, :]
            z = Xa_p[component_z, :]

            # apply some smoothing to the trajectories
            x = gaussian_filter1d(x, sigma=sigma)
            y = gaussian_filter1d(y, sigma=sigma)
            z = gaussian_filter1d(z, sigma=sigma)

            ax.plot(x, y, z, c='darkturquoise')

            # plot dots at initial point
            ax.scatter(x[0], y[0], z[0], c='darkturquoise', s=14)

            # make the axes a bit cleaner
            style_3d_ax(ax)

    # specify the orientation of the 3d plot
    ax1.view_init(elev=22, azim=30)
    ax2.view_init(elev=22, azim=110)

    plt.suptitle(f"{mouse_id} {day}")
    plt.tight_layout()
    plt.show()


#%%

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
    data_file : str, np.ndarray, nwbfile
        A string pointing to the file to be used for data analysis, or a numpy array, or a neurodata without borders file.

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


#%% Test PCA from neuronal_dynamics_projection.py
from neuronal_dynamics_projection import NeuronalDynamicsProjection