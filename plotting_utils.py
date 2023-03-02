"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 04-30-2022
File Final Edit Date:

Description: Plotting utilities.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy

#%% Plotting style
#Todo: add to dg_network_graph
pal = sns.color_palette('husl', 9)
# plt.style.use('/nng.mplstyle') #Commmented out because BLA analysis is trying to use this and cannot find the reference

def plot_matched_data(set1, set2, labels, colors):

    # Put into dataframe
    df = pd.DataFrame({labels[0]: set1, labels[1]: set2})
    data = pd.melt(df)

    # Plot
    fig, ax = plt.subplots()
    sns.swarmplot(data=data, x='variable', y='value', ax=ax, size=0)

    # Find idx0 and idx1 by inspecting the elements returned from ax.get_children()
    idx0 = 0
    idx1 = 1
    locs1 = ax.get_children()[idx0].get_offsets()
    locs2 = ax.get_children()[idx1].get_offsets()

    y_all = np.zeros(2)
    for i in range(locs1.shape[0]):
        x = [locs1[i, 0], locs2[i, 0]]
        y = [locs1[i, 1], locs2[i, 1]]
        ax.plot(x, y, color='lightgrey', linewidth=0.5)
        ax.plot(locs1[i, 0], locs1[i, 1], '.', color=colors[0])
        ax.plot(locs2[i, 0], locs2[i, 1], '.', color=colors[1])
        data = [locs1[:, 1], locs2[:, 1]]
        ax.boxplot(data, positions=[0, 1], capprops=dict(linewidth=0.5, color='k'),
                   whiskerprops=dict(linewidth=0.5, color='k'),
                   boxprops=dict(linewidth=0.5, color='k'),
                   medianprops=dict(color='k'))
        plt.xticks([])
        y_all = np.vstack((y_all, y))

    plt.xlabel(f'P-value = {scipy.stats.ttest_rel(set1, set2).pvalue:.3}')

