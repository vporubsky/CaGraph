"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: July 9, 2021
File Final Edit Date:

Description: An initial investigation of the prazosin dataset, which includes
exposure to a novel context C after days 1-9 of fear conditioning.

Experimental set-up:
- Recordings taken on D1 and D9 in Context A and B as before
- Recordings taken on D10 and D12 in Context C
- Separate groups that are given saline + CNO or prazosin + CNO

First dataset animals:
- 198-1
- 198-3
- 202-2
- 202-4 (prazosin)

Experimental conclusions from Eric:
- low number of neurons in the field of view
- freezing behavior doesn't suggest that the mice can descriminate between the two contexts
- neither saline or prazosin discriminate between A and B well
- the prazosin group has higher freezing on D9 but conclusions not clear (low n)
"""
from neuronal_network_graph import DGNetworkGraph as nng
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

#%% Global analysis parameters
threshold = 0.3

#%% Load untreated data files - saline
files = os.listdir()

# Clustering Coefficient
cc_A_D1 = []
cc_A_D9 = []
cc_B_D1 = []
cc_B_D9 = []
cc_D10 = []
cc_D12 = []

# Correlated Pairs Ratio
cr_A_D1 = []
cr_A_D9 = []
cr_B_D1 = []
cr_B_D9 = []
cr_D10 = []
cr_D12 = []
for ca_file in files:


    if ca_file.find('D1_') != -1:
        mouse_id = ca_file[0:5]

        nn = nng(ca_file)
        print(f"Executing analyses for {mouse_id}")
        num_neurons = nn.num_neurons

        # Context A and B graphs
        conA = nn.get_context_A_graph(threshold=threshold)
        conB = nn.get_context_B_graph(threshold=threshold)

        # clustering coefficient
        cc_A = nn.get_context_A_clustering_coefficient()
        cc_B = nn.get_context_B_clustering_coefficient()

        # correlated pairs ratio
        cr_A = nn.get_context_A_correlated_pair_ratio(threshold=threshold)
        cr_B = nn.get_context_B_correlated_pair_ratio(threshold=threshold)

        cc_A_D1.append(cc_A)
        cc_B_D1.append(cc_B)
        cr_A_D1.append(cr_A)
        cr_B_D1.append(cr_B)

    if ca_file.find('D9_') != -1:
        mouse_id = ca_file[0:5]

        nn = nng(ca_file)
        print(f"Executing analyses for {mouse_id}")
        num_neurons = nn.num_neurons

        # Context A and B graphs
        conA = nn.get_context_A_graph(threshold=threshold)
        conB = nn.get_context_B_graph(threshold=threshold)

        # clustering coefficient
        cc_A = nn.get_context_A_clustering_coefficient()
        cc_B = nn.get_context_B_clustering_coefficient()

        # correlated pairs ratio
        cr_A = nn.get_context_A_correlated_pair_ratio(threshold=threshold)
        cr_B = nn.get_context_B_correlated_pair_ratio(threshold=threshold)

        cc_A_D9.append(cc_A)
        cc_B_D9.append(cc_B)
        cr_A_D9.append(cr_A)
        cr_B_D9.append(cr_B)

    if ca_file.find('D10C') != -1:
        mouse_id = ca_file[0:5]

        nn = nng(ca_file)
        print(f"Executing analyses for {mouse_id}")
        num_neurons = nn.num_neurons

        G = nn.get_network_graph(threshold=threshold)

        # clustering coefficient
        cc = nn.get_clustering_coefficient(threshold=threshold, graph=G)

        # correlated pairs ratio
        cr = nn.get_correlated_pair_ratio(threshold=threshold, graph=G)

        cc_D10.append(cc)
        cr_D10.append(cr)

    if ca_file.find('D12C') != -1:
        mouse_id = ca_file[0:5]

        nn = nng(ca_file)
        print(f"Executing analyses for {mouse_id}")
        num_neurons = nn.num_neurons

        G = nn.get_network_graph(threshold=threshold)

        # clustering coefficient
        cc = nn.get_clustering_coefficient(threshold=threshold, graph=G)

        # correlated pairs ratio
        cr = nn.get_correlated_pair_ratio(threshold=threshold, graph=G)

        cc_D12.append(cc)
        cr_D12.append(cr)

#%% Load untreated data files - prazosin
os.chdir(os.getcwd() + '/prazosin')
files = os.listdir()

# Clustering Coefficient
cc_A_D1_praz = []
cc_A_D9_praz = []
cc_B_D1_praz = []
cc_B_D9_praz = []
cc_D10_praz = []
cc_D12_praz = []

# Correlated Pairs Ratio
cr_A_D1_praz = []
cr_A_D9_praz = []
cr_B_D1_praz = []
cr_B_D9_praz = []
cr_D10_praz = []
cr_D12_praz = []

for ca_file in files:
    if ca_file.find('D1_') != -1:
        mouse_id = ca_file[0:5]

        nn = nng(ca_file)
        print(f"Executing analyses for {mouse_id}")
        num_neurons = nn.num_neurons

        # Context A and B graphs
        conA = nn.get_context_A_graph(threshold=threshold)
        conB = nn.get_context_B_graph(threshold=threshold)

        # clustering coefficient
        cc_A = nn.get_context_A_clustering_coefficient()
        cc_B = nn.get_context_B_clustering_coefficient()

        # correlated pairs ratio
        cr_A = nn.get_context_A_correlated_pair_ratio(threshold=threshold)
        cr_B = nn.get_context_B_correlated_pair_ratio(threshold=threshold)

        cc_A_D1_praz.append(cc_A)
        cc_B_D1_praz.append(cc_B)
        cr_A_D1_praz.append(cr_A)
        cr_B_D1_praz.append(cr_B)

    if ca_file.find('D9_') != -1:
        mouse_id = ca_file[0:5]

        nn = nng(ca_file)
        print(f"Executing analyses for {mouse_id}")
        num_neurons = nn.num_neurons

        # Context A and B graphs
        conA = nn.get_context_A_graph(threshold=threshold)
        conB = nn.get_context_B_graph(threshold=threshold)

        # clustering coefficient
        cc_A = nn.get_context_A_clustering_coefficient()
        cc_B = nn.get_context_B_clustering_coefficient()

        # correlated pairs ratio
        cr_A = nn.get_context_A_correlated_pair_ratio(threshold=threshold)
        cr_B = nn.get_context_B_correlated_pair_ratio(threshold=threshold)

        cc_A_D9_praz.append(cc_A)
        cc_B_D9_praz.append(cc_B)
        cr_A_D9_praz.append(cr_A)
        cr_B_D9_praz.append(cr_B)

    if ca_file.find('D10C') != -1:
        mouse_id = ca_file[0:5]

        nn = nng(ca_file)
        print(f"Executing analyses for {mouse_id}")
        num_neurons = nn.num_neurons

        G = nn.get_network_graph(threshold=threshold)

        # clustering coefficient
        cc = nn.get_clustering_coefficient(threshold=threshold, graph=G)

        # correlated pairs ratio
        cr = nn.get_correlated_pair_ratio(threshold=threshold, graph=G)

        cc_D10_praz.append(cc)
        cr_D10_praz.append(cr)

    if ca_file.find('D12C') != -1:
        mouse_id = ca_file[0:5]

        nn = nng(ca_file)
        print(f"Executing analyses for {mouse_id}")
        num_neurons = nn.num_neurons

        G = nn.get_network_graph(threshold=threshold)

        # clustering coefficient
        cc = nn.get_clustering_coefficient(threshold=threshold, graph=G)

        # correlated pairs ratio
        cr = nn.get_correlated_pair_ratio(threshold=threshold, graph=G)

        cc_D12_praz.append(cc)
        cr_D12_praz.append(cr)

#%% TMP: not enough data, only use one prazosin and one saline
saline_idx = 2
praz_idx = 0

#%% Figure 1
# CDF prestim, stim ----------------------------------------------
stat_lev = stats.ks_2samp(cc_A_D1[saline_idx], cc_A_D9[saline_idx])

# sort the data in ascending order
x = np.sort(cc_A_D1[saline_idx])
# get the cdf values of y
y = np.arange(len(cc_A_D1[saline_idx])) / float(len(cc_A_D1[saline_idx]))

 # plotting
plt.figure(figsize=(15, 15))
plt.subplot(221)
plt.plot(x, y, 'salmon', marker='o')


# sort the data in ascending order
x = np.sort(cc_B_D1[saline_idx])
# get the cdf values of y
y = np.arange(len(cc_B_D1[saline_idx])) / float(len(cc_B_D1[saline_idx]))

 # plotting
plt.subplot(221)
plt.plot(x, y, 'turquoise', marker='o')


# sort the data in ascending order
x = np.sort(cc_B_D1_praz[praz_idx])
# get the cdf values of y
y = np.arange(len(cc_B_D1_praz[praz_idx])) / float(len(cc_B_D1_praz[praz_idx]))

# plotting
plt.subplot(221)
plt.plot(x, y, 'grey', marker='o')

# sort the data in ascending order
x = np.sort(cc_A_D1_praz[praz_idx])
# get the cdf values of y
y = np.arange(len(cc_A_D1_praz[praz_idx])) / float(len(cc_A_D1_praz[praz_idx]))

# plotting
plt.subplot(221)
plt.plot(x, y, 'lightgrey', marker='o')

plt.title('D1')



# CDF prestim, stim ----------------------------------------------
saline_idx = 0
# sort the data in ascending order
x = np.sort(cc_A_D9[saline_idx])
# get the cdf values of y
y = np.arange(len(cc_A_D9[saline_idx])) / float(len(cc_A_D9[saline_idx]))

 # plotting
plt.subplot(222)
plt.plot(x, y, 'salmon', marker='o')


# sort the data in ascending order
x = np.sort(cc_B_D9[saline_idx])
# get the cdf values of y
y = np.arange(len(cc_B_D9[saline_idx])) / float(len(cc_B_D9[saline_idx]))

 # plotting
plt.subplot(222)
plt.plot(x, y, 'turquoise', marker='o')


# sort the data in ascending order
x = np.sort(cc_B_D9_praz[praz_idx])
# get the cdf values of y
y = np.arange(len(cc_B_D9_praz[praz_idx])) / float(len(cc_B_D9_praz[praz_idx]))

# plotting
plt.subplot(222)
plt.plot(x, y, 'grey', marker='o')

# sort the data in ascending order
x = np.sort(cc_A_D9_praz[praz_idx])
# get the cdf values of y
y = np.arange(len(cc_A_D9_praz[praz_idx])) / float(len(cc_A_D9_praz[praz_idx]))

# plotting
plt.subplot(222)
plt.plot(x, y, 'lightgrey', marker='o')

plt.title('D9')



# CDF prestim, stim ----------------------------------------------
saline_idx = 2
# sort the data in ascending order
x = np.sort(cc_D10[saline_idx])
# get the cdf values of y
y = np.arange(len(cc_D10[saline_idx])) / float(len(cc_D10[saline_idx]))

 # plotting
plt.subplot(223)
plt.plot(x, y, 'lightblue', marker='o')



# sort the data in ascending order
x = np.sort(cc_D10_praz[praz_idx])
# get the cdf values of y
y = np.arange(len(cc_D10_praz[praz_idx])) / float(len(cc_D10_praz[praz_idx]))

# plotting
plt.subplot(223)
plt.plot(x, y, 'grey', marker='o')


plt.title('D10 context C')


# CDF prestim, stim ----------------------------------------------
saline_idx = 2
# sort the data in ascending order
x = np.sort(cc_D12[saline_idx])
# get the cdf values of y
y = np.arange(len(cc_D12[saline_idx])) / float(len(cc_D12[saline_idx]))

 # plotting
plt.subplot(224)
plt.plot(x, y, 'lightblue', marker='o')



# sort the data in ascending order
x = np.sort(cc_D12_praz[praz_idx])
# get the cdf values of y
y = np.arange(len(cc_D12_praz[praz_idx])) / float(len(cc_D12_praz[praz_idx]))

# plotting
plt.subplot(224)
plt.plot(x, y, 'grey', marker='o')


plt.title('D12 context C')

plt.savefig('tmp_1.png', dpi=300)
plt.show()




#%% Figure 2 Histograms
plt.figure(figsize=(15, 15))

# sort the data in ascending order
x = np.sort(cc_A_D1[saline_idx])
 # plotting
plt.subplot(221)
plt.hist(x, color='salmon', alpha=0.5, bins=20)


# sort the data in ascending order
x = np.sort(cc_B_D1[saline_idx])
 # plotting
plt.subplot(221)
plt.hist(x, color='turquoise', alpha=0.5, bins=20)
plt.xlim((0,1))

# sort the data in ascending order
x = np.sort(cc_B_D1_praz[praz_idx])
 # plotting
plt.subplot(221)
plt.hist(x, color='grey', alpha=0.4, bins=20)

# sort the data in ascending order
x = np.sort(cc_A_D1_praz[praz_idx])
 # plotting
plt.subplot(221)
plt.hist(x, color='lightgrey', alpha=0.4, bins=20)

plt.title('D1')



# CDF prestim, stim ----------------------------------------------
saline_idx = 0
# sort the data in ascending order
x = np.sort(cc_A_D9[saline_idx])
# plotting
plt.subplot(222)
plt.xlim((0,1))
plt.hist(x, color='salmon', alpha=0.5, bins=20)


# sort the data in ascending order
x = np.sort(cc_B_D9[saline_idx])
# plotting
plt.subplot(222)
plt.hist(x, color='turquoise', alpha=0.5, bins=20)


# sort the data in ascending order
x = np.sort(cc_B_D9_praz[praz_idx])
# plotting
plt.subplot(222)
plt.hist(x, color='grey', alpha=0.4, bins=20)

# sort the data in ascending order
x = np.sort(cc_A_D9_praz[praz_idx])
# plotting
plt.subplot(222)
plt.hist(x, color='lightgrey', alpha=0.4, bins=20)

plt.title('D9')


saline_idx = 2
# sort the data in ascending order
x = np.sort(cc_D10[saline_idx])
# plotting
plt.subplot(223)
plt.xlim((0,1))
plt.hist(x, color='lightblue', alpha=0.5, bins=20)

# sort the data in ascending order
x = np.sort(cc_D10_praz[praz_idx])
# plotting
plt.subplot(223)
plt.hist(x, color='grey', alpha=0.4, bins=20)


plt.title('D10 context C')

saline_idx = 2
# sort the data in ascending order
x = np.sort(cc_D12[saline_idx])
 # plotting
plt.subplot(224)
plt.xlim((0,1))
plt.hist(x, color='lightblue', alpha=0.5, bins=20)



# sort the data in ascending order
x = np.sort(cc_D12_praz[praz_idx])
# plotting
plt.subplot(224)
plt.hist(x, color='grey', alpha=0.4, bins=20)


plt.title('D12 context C')

plt.savefig('tmp_2.png', dpi=300)
plt.show()




#%% TMP: not enough data, only use one prazosin and one saline
saline_idx = 2
praz_idx = 0

#%% Figure 3
# sort the data in ascending order
x = np.sort(cc_A_D1[saline_idx])
# get the cdf values of y
y = np.arange(len(cc_A_D1[saline_idx])) / float(len(cc_A_D1[saline_idx]))

 # plotting
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(x, y, 'salmon', marker='o')


# sort the data in ascending order
x = np.sort(cc_B_D1[saline_idx])
# get the cdf values of y
y = np.arange(len(cc_B_D1[saline_idx])) / float(len(cc_B_D1[saline_idx]))

 # plotting
plt.subplot(121)
plt.plot(x, y, 'turquoise', marker='o')


# sort the data in ascending order
x = np.sort(cc_D10[saline_idx])
# get the cdf values of y
y = np.arange(len(cc_D10[saline_idx])) / float(len(cc_D10[saline_idx]))

 # plotting
plt.subplot(121)
plt.plot(x, y, 'lightblue', marker='o')

plt.title('Saline')


# sort the data in ascending order
x = np.sort(cc_B_D1_praz[praz_idx])
# get the cdf values of y
y = np.arange(len(cc_B_D1_praz[praz_idx])) / float(len(cc_B_D1_praz[praz_idx]))

# plotting
plt.subplot(122)
plt.plot(x, y, 'turquoise', marker='o')

# sort the data in ascending order
x = np.sort(cc_A_D1_praz[praz_idx])
# get the cdf values of y
y = np.arange(len(cc_A_D1_praz[praz_idx])) / float(len(cc_A_D1_praz[praz_idx]))

# plotting
plt.subplot(122)
plt.plot(x, y, 'salmon', marker='o')



# sort the data in ascending order
x = np.sort(cc_D10_praz[praz_idx])
# get the cdf values of y
y = np.arange(len(cc_D10_praz[praz_idx])) / float(len(cc_D10_praz[praz_idx]))

 # plotting
plt.subplot(122)
plt.plot(x, y, 'lightblue', marker='o')


plt.title('Prazosin')
plt.savefig('tmp3.png', dpi=300)
plt.show()
