# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:06:51 2020

@author: Veronica Porubsky

Title: Dimensionality reduction studies
"""
from neuronalNetworkGraph import neuronalNetworkGraph
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 3D plotting
from sklearn.decomposition import PCA # use to perform decomposition
from sklearn.preprocessing import StandardScaler # use to normalize data before PCA
import pandas as pd # use to arrange data into dataframes
import numpy as np

nn = neuronalNetworkGraph('mouse_7_with_treatment_day_9_all_calcium_traces.npy')
mat = nn.neuron_dynamics
df = pd.DataFrame(data = mat)

# Normalize the features
x = StandardScaler().fit_transform(df)

# perform PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

## visualize components in 2D
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(111)
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# ax.scatter(principalDf.loc[:, 'principal component 1'], principalDf.loc[:, 'principal component 2'])
# ax.grid()

## visualize components in 3D
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(111, projection='3d') 
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_ylabel('Principal Component 3', fontsize = 15)
# ax.set_title('3 component PCA', fontsize = 20)
# ax.scatter(principalDf.loc[:, 'principal component 1'], principalDf.loc[:, 'principal component 2'], principalDf.loc[:, 'principal component 3'])
# ax.grid()

#%% Arrange data to perform PCA on all mice, all days
names = []
targets = []
for i in range(1, 11):
    for j in [1, 9]:
        filename = 'mouse_' + str(i) +'_with_treatment_day_' + str(j) + '_all_calcium_traces.npy'
        nn = neuronalNetworkGraph(filename)
        data_mat_curr = nn.neuron_dynamics
        for k in range(np.shape(data_mat_curr)[0]):
            names.append('mouse_' + str(i) +'_day_' + str(j) + '_neuron_' + str(k))
            targets.append('mouse_' + str(i))
        if i == 1 and j == 1:
            data_mat = data_mat_curr
        else:
            data_mat = np.vstack((data_mat, data_mat_curr))
data_mat = data_mat     


df = pd.DataFrame(data = data_mat)
target_df = pd.DataFrame(data = targets, columns = ['target'])      
x = StandardScaler().fit_transform(df)
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
finalDf = pd.concat([principalDf, target_df], axis = 1)


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111)
ax.set_xlabel('Principal Component 2', fontsize = 15)
ax.set_ylabel('Principal Component 3', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

target_categories = ['mouse_1', 'mouse_2', 'mouse_3', 'mouse_4', 'mouse_5', 'mouse_6', 'mouse_7', 'mouse_8', 'mouse_9', 'mouse_10']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
for target, color in zip(target_categories, colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s = 50)
ax.legend(target_categories)
ax.grid()


#%% Arrange data as before, but also arrange with contexts split (time = 0:1800 and time = 1800:3600)





#%% test plotting of timecourse data
# time = np.linspace(0, 360, 3600)
# plt.figure(1)
# plt.plot(time, mat[0, :])
# plt.figure(2)
# plt.plot(time, df.loc[0, :])
# plt.figure(3)
# plt.plot(time, x[1, :])



#%% Plot figures side by side day 1, day 9 for each mouse
for i in range(1, 11):
    plt.clf()
    figname = 'mouse_' + str(i) +'_with_treatment_all_calcium_traces_PCA.png'
    for j in [1, 9]:
        filename = 'mouse_' + str(i) +'_with_treatment_day_' + str(j) + '_all_calcium_traces.npy'
        nn = neuronalNetworkGraph(filename)
        mat = nn.neuron_dynamics
        df = pd.DataFrame(data = mat)
        x = StandardScaler().fit_transform(df)        
        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
        fig = plt.figure(1, figsize = (30,8))
        if j == 1:
            ax = fig.add_subplot(121)
            ax.set_ylabel('Principal Component 2')
        else:
            ax = fig.add_subplot(122)
            ax.set_yticklabels([])
        ax.set_xlabel('Principal Component 1')
        ax.set_title('Day ' + str(j), fontsize = 12)
        ax.scatter(principalDf.loc[:, 'principal component 1'], principalDf.loc[:, 'principal component 2'])
        ax.set_xlim([-100, 250])
        ax.set_ylim([-100, 150])
        ax.set
        ax.grid()
    # fig.xlabel('Principal Component 1')
    # fig.ylabel('Principal Component 2')
    fig.suptitle('Mouse ' + str(i) + ' PCA', fontsize = 20)
    plt.savefig(figname, dpi = 300)
    
    