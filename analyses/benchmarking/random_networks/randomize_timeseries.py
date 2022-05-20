"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 05-11-2022
File Final Edit Date:

Description: methods to generate randomized timeseries from neural data.
"""
# Import packages
import os

from setup import FC_DATA_PATH
from dg_network_graph import DGNetworkGraph as nng
import numpy as np
import matplotlib.pyplot as plt
import sklearn

RANDOM_DATA_PATH = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/analyses/benchmarking/random_networks/randomized_neural_data/'
EXPORT_PATH = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/analyses/benchmarking/random_networks/'
# how many datasets do I need to use to generate a random sample? Should I just take all datasets in and create a new one?
def bins(lst, n):
    """Yield successive n-sized chunks from lst."""
    lst = list(lst)
    build_binned_list = []
    for i in range(0, len(lst), n):
        build_binned_list.append(lst[i:i + n])
    return build_binned_list

# Function definitions
def generate_randomized_timeseries_matrix(data: list) -> np.ndarray:
    """
    data: list

    Parameter data should contain a list of np.ndarray objects.

    Return a numpy array or NWB file.
    """
    time = data[0, :].copy()
    for row in range(np.shape(data)[0]):
        np.random.shuffle(data[row, :])
    data[0, :] = time.copy()
    return data

def generate_randomized_timeseries_binned(data: list, bin_size: int) -> np.ndarray:
    """
    data: list

    Parameter data should contain a list of np.ndarray objects.

    Return a numpy array or NWB file.
    """
    time = data[0, :].copy()
    build_new_array = np.array(bins(lst=data[1, :], n=bin_size))

    # build binned dist
    for row in range(np.shape(data[2:, :])[0]):
        binned_row = bins(lst=data[row + 2, :], n=bin_size)
        build_new_array = np.vstack([build_new_array, binned_row])

    for row in range(np.shape(build_new_array)[0]):
        np.random.shuffle(build_new_array[row, :])

    flatten_array = time.copy()
    for row in range(np.shape(build_new_array)[0]):
        flat_row = [item for sublist in build_new_array[row, :] for item in sublist]
        flatten_array = np.vstack([flatten_array, flat_row])

    return flatten_array

#%%
data = np.genfromtxt(FC_DATA_PATH + '/2-1_D1_smoothed_calcium_traces.csv', delimiter=',')
random_binned_data = generate_randomized_timeseries_binned(data=data.copy(), bin_size=20)
random_data = generate_randomized_timeseries_matrix(data=data.copy())
# Check if context A only random networks are different than context B only, and how they compare to

plt.plot(random_binned_data[0, :], random_binned_data[1,:])
plt.title('Binned random data')
plt.show()
np.savetxt(RANDOM_DATA_PATH + '2-1_random_binned_data.csv', random_binned_data, delimiter=",")
np.savetxt('2-1_random_data.csv', random_data, delimiter=",")

#%%
plt.figure()
plt.subplot(211)
plt.plot(data[0, :], data[4, :], c = 'blue', label='ground truth')
plt.ylabel('ΔF/F')
plt.legend()
plt.subplot(212)
plt.plot(random_binned_data[0, :], random_binned_data[4, :], c='grey', label='shuffled')
plt.ylabel('')
plt.ylabel('ΔF/F')
plt.xlabel('Time')
plt.legend()
plt.savefig(EXPORT_PATH + 'timeseries_randomized.png', dpi=200)
plt.show()


#%% Random
random_nng = nng(RANDOM_DATA_PATH + '2-1_random_binned_data.csv')
x=random_nng.pearsons_correlation_matrix
np.fill_diagonal(x, 0)
heatmap = plt.imshow(x, cmap='hot', interpolation='nearest')
plt.colorbar(heatmap)
plt.savefig(EXPORT_PATH + 'random_heatmap.png', dpi=200)
plt.show()

print(f"The median Pearson's r-value is: {np.median(x)}")
print(f"The mean Pearson's r-value is: {np.mean(x)}")
print(f"The max Pearson's r-value is: {np.max(x)}")


#%% Normal

random_nng = nng(FC_DATA_PATH + '/2-1_D1_smoothed_calcium_traces.csv')
y=random_nng.pearsons_correlation_matrix
np.fill_diagonal(y, 0)
heatmap = plt.imshow(y, cmap='hot', interpolation='nearest')
plt.colorbar(heatmap)
plt.savefig(EXPORT_PATH + 'ground_truth_heatmap.png', dpi=200)
plt.show()

print(f"The median Pearson's r-value is: {np.median(y)}")
print(f"The mean Pearson's r-value is: {np.mean(y)}")
print(f"The max Pearson's r-value is: {np.max(y)}")

#%% Context A
random_nng = nng(FC_DATA_PATH + '/2-1_D1_smoothed_calcium_traces.csv')
z=random_nng.con_A_pearsons_correlation_matrix
np.fill_diagonal(z, 0)
heatmap = plt.imshow(z, cmap='hot', interpolation='nearest')
plt.colorbar(heatmap)
plt.savefig(EXPORT_PATH + 'context_A_heatmap.png', dpi=200)
plt.show()

print(f"The median Pearson's r-value is: {np.median(z)}")
print(f"The mean Pearson's r-value is: {np.mean(z)}")
print(f"The max Pearson's r-value is: {np.max(z)}")

#%% Context B
random_nng = nng(FC_DATA_PATH + '/2-1_D1_smoothed_calcium_traces.csv')
b=random_nng.con_B_pearsons_correlation_matrix
np.fill_diagonal(b, 0)
heatmap = plt.imshow(b, cmap='hot', interpolation='nearest')
plt.colorbar(heatmap)
plt.savefig(EXPORT_PATH + 'context_B_heatmap.png', dpi=200)
plt.show()

print(f"The median Pearson's r-value is: {np.median(b)}")
print(f"The mean Pearson's r-value is: {np.mean(b)}")
print(f"The max Pearson's r-value is: {np.max(b)}")


#%%
plt.hist(np.tril(x).flatten(), bins = 50, color='grey', alpha = 0.3)
plt.hist(np.tril(y).flatten(), bins = 50, color= 'blue', alpha = 0.3)


plt.ylim((0,200))
plt.legend(['shuffled', 'ground truth'])
plt.savefig(EXPORT_PATH + 'random_v_ground_truth_hist.png', dpi=200)
plt.show()

#%%
plt.hist(np.tril(x).flatten(), bins = 50, color='grey', alpha = 0.3)
plt.hist(np.tril(z).flatten(), bins = 50, color= 'salmon', alpha = 0.3)

plt.ylim((0,200))
plt.legend(['shuffled', 'context A'])
plt.savefig(EXPORT_PATH + 'random_v_context_A_hist.png', dpi=200)
plt.show()

#%%

plt.hist(np.tril(x).flatten(), bins = 50, color='grey', alpha = 0.3)
plt.hist(np.tril(b).flatten(), bins = 50, color='teal', alpha = 0.3)

plt.ylim((0,200))
plt.legend(['shuffled', 'context B'])
plt.savefig(EXPORT_PATH + 'random_v_context_B_hist.png', dpi=200)
plt.show()