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
# Wildtype condition
D1_WT = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv',
         '1055-4_D1_smoothed_calcium_traces.csv',
         '122-1_D1_smoothed_calcium_traces.csv',
         '122-2_D1_smoothed_calcium_traces.csv', '122-3_D1_smoothed_calcium_traces.csv', '14-0_D1_smoothed_calcium_traces.csv']
D9_WT = ['1055-1_D9_smoothed_calcium_traces.csv', '1055-2_D9_smoothed_calcium_traces.csv',
         '1055-4_D9_smoothed_calcium_traces.csv',
         '14-0_D9_smoothed_calcium_traces.csv', '122-1_D9_smoothed_calcium_traces.csv',
         '122-2_D9_smoothed_calcium_traces.csv', '122-3_D9_smoothed_calcium_traces.csv']


path_to_data = "/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/LC-DG-FC-data/"

data_file = D1_WT[3]
mouse_id = data_file.replace('_smoothed_calcium_traces.csv', '')
nn = nng(path_to_data + data_file)

#%% Normalize Data
# Todo: normalize as necessary

#%% Compute covariance matrix
data = nn.neuron_dynamics
cov = (1/len(data))*np.dot(data,np.transpose(data))
plt.imshow(cov, cmap='hot')
plt.show()

#%% Eigenvalue decomposition
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

#%% Singular value decomposition in two dimensions
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
data_A-=np.mean(data_A)
data_A/=np.std(data_A)
data_B = np.transpose(nn.context_B_dynamics)
data_B-=np.mean(data_B)
data_B/=np.std(data_B)

# SVD
U_a, Sig_a, V_a = svd(data_A)
U_b, Sig_b, V_b = svd(data_B)

# Project dataset to three dimensions
V_r_a = V_a[:, 0:2]
V_r_b = V_b[:, 0:2]
projection_a = np.matmul(data_A, V_r_a)
projection_b = np.matmul(data_B, V_r_b)

# Plot projection/ trajectory in state space
plt.title(f"{mouse_id} on {day}")
plt.plot(projection_b[:,0], projection_b[:,1], '-', c = 'teal', alpha=0.5);
plt.plot(projection_a[:,0], projection_a[:,1], '-', c = 'salmon', alpha=0.5);
plt.show()
#%% SVD in three dimensions
V_r_a = V_a[:, 0:3]
V_r_b = V_b[:, 0:3]
projection_a = np.matmul(data_A, V_r_a)
projection_b = np.matmul(data_B, V_r_b)
# Plot projection/ trajectory in state space
ax = plt.axes(projection='3d')
ax.scatter3D(projection_a[:,0], projection_a[:,1], projection_a[:,2], '.', c='salmon');
ax.scatter3D(projection_b[:,0], projection_b[:,1], projection_b[:,2], '.', c='teal');
plt.show()


#%% Singular value decomposition in two dimensions
# Todo: check normalization
ax = plt.axes(projection='3d')
mouse_idx = 6
data_file = D1_WT[mouse_idx]


mouse_id = data_file.replace('_smoothed_calcium_traces.csv', '')
nn = nng(path_to_data + data_file)

data_A = np.transpose(nn.context_A_dynamics)
data_A-=np.mean(data_A)
data_A/=np.std(data_A)
data_B = np.transpose(nn.context_B_dynamics)
data_B-=np.mean(data_B)
data_B/=np.std(data_B)

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
data_A-=np.mean(data_A)
data_A/=np.std(data_A)
data_B = np.transpose(nn.context_B_dynamics)
data_B-=np.mean(data_B)
data_B/=np.std(data_B)

# SVD
U_a, Sig_a, V_a = svd(data_A)
U_b, Sig_b, V_b = svd(data_B)

V_r_a = V_a[:, 0:3]
V_r_b = V_b[:, 0:3]
projection_a9 = np.matmul(data_A, V_r_a)
projection_b9 = np.matmul(data_B, V_r_b)

# Plot projection/ trajectory in state space

ax.scatter3D(projection_a1[:,0], projection_a1[:,1], projection_a1[:,2], '.', color='salmon');
ax.scatter3D(projection_b1[:,0], projection_b1[:,1], projection_b1[:,2], '.', color='teal');

ax.scatter3D(projection_a9[:,0], projection_a9[:,1], projection_a9[:,2], '.', color='mistyrose');
ax.scatter3D(projection_b9[:,0], projection_b9[:,1], projection_b9[:,2], '.', color='turquoise');
plt.show()


#%% Singular value decomposition in two dimensions with plotly express
import plotly.express as px
ax = plt.axes(projection='3d')
mouse_idx = 6
data_file = D1_WT[mouse_idx]


mouse_id = data_file.replace('_smoothed_calcium_traces.csv', '')
nn = nng(path_to_data + data_file)

data_A = np.transpose(nn.context_A_dynamics)
data_A-=np.mean(data_A)
data_A/=np.std(data_A)
data_B = np.transpose(nn.context_B_dynamics)
data_B-=np.mean(data_B)
data_B/=np.std(data_B)

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
data_A-=np.mean(data_A)
data_A/=np.std(data_A)
data_B = np.transpose(nn.context_B_dynamics)
data_B-=np.mean(data_B)
data_B/=np.std(data_B)

# SVD
U_a, Sig_a, V_a = svd(data_A)
U_b, Sig_b, V_b = svd(data_B)

V_r_a = V_a[:, 0:3]
V_r_b = V_b[:, 0:3]
projection_a9 = np.matmul(data_A, V_r_a)
projection_b9 = np.matmul(data_B, V_r_b)

data_a1 = [[projection_a1[:, 0], projection_a1[:, 1], projection_a1[:,2]]]
df_a1 = pd.DataFrame(data_a1, columns = ['PC1', 'PC2', 'PC3'])
print(df_a1)

fig = px.scatter_3d(df_a1, x='PC1', y='PC2', z='PC3', color='PC1', symbol ='PC1')
fig.show
fig.show()
# Plot projection/ trajectory in state space

ax.scatter3D(projection_a1[:,0], projection_a1[:,1], projection_a1[:,2], '.', color='salmon');
ax.scatter3D(projection_b1[:,0], projection_b1[:,1], projection_b1[:,2], '.', color='teal');

ax.scatter3D(projection_a9[:,0], projection_a9[:,1], projection_a9[:,2], '.', color='mistyrose');
ax.scatter3D(projection_b9[:,0], projection_b9[:,1], projection_b9[:,2], '.', color='turquoise');
plt.show()

#%% Look at only cells that are preferentially active in a given context