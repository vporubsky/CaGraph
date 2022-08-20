"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 
File Final Edit Date:

Description: 
"""
# Import packages
from setup import FC_DATA_PATH
from dg_network_graph import DGNetworkGraph as nng

path_to_data = FC_DATA_PATH
import numpy as np
import matplotlib.pyplot as plt

RANDOM_DATA_PATH = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/analyses/benchmarking/random_networks/randomized_neural_data/'
EXPORT_PATH = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/analyses/benchmarking/random_networks/'

def bins(lst, n):
    """Yield successive n-sized chunks from lst."""
    lst = list(lst)
    build_binned_list = []
    for i in range(0, len(lst), n):
        build_binned_list.append(lst[i:i + n])
    return build_binned_list

def event_bins(data, events):
    """
    :param data:
    :param events: single events timecourse
    :return:
    """
    data = list(data)
    build_binned_list = []
    event_idx = list(np.nonzero(events)[0])
    if event_idx[-1] != len(data):
        event_idx.append(len(data))
    start_val = 0
    for idx in event_idx:
        build_binned_list.append(data[start_val:idx])
        start_val = idx
    np.random.shuffle(build_binned_list)
    flat_random_binned_list = [item for sublist in build_binned_list for item in sublist]
    return flat_random_binned_list


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

def generate_event_segmented(data: list, event_data: list) -> np.ndarray:
    """
    data: list

    Parameter data should contain a list of np.ndarray objects.

    Return a numpy array or NWB file.
    """
    time = data[0, :].copy()
    build_new_array = np.array(event_bins(data=data[1, :], events=event_data[1,:]))

    # build binned dist
    flatten_array = time.copy()
    for row in range(np.shape(data[2:, :])[0]):
        binned_row = event_bins(data=data[row + 2, :], events = event_data[row + 2, :])
        flatten_array = np.vstack([flatten_array, binned_row])

    return flatten_array



#%% Demonstrate binned randomization --rerun
file_str = '2-1'
day = 'D9'
if day == 'D1':
    event_day = 'Day1'
else:
    event_day = 'Day9'
data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')
event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')
random_event_binned_data = generate_event_segmented(data=data.copy(), event_data=event_data)


plt.plot(event_data[0,:], event_data[1,:], '.')
# Check if context A only random networks are different than context B only, and how they compare to
plt.plot(random_event_binned_data[0, :], random_event_binned_data[1,:], color='grey', alpha=0.5)
plt.plot(data[0, 0:1800], data[1,0:1800], 'darkturquoise', alpha = 0.5)
plt.plot(data[0, 1800:3600], data[1,1800:3600], 'salmon', alpha = 0.5)
plt.title('Binned random data')
plt.show()


random_nng = nng(random_event_binned_data)
x = random_nng.pearsons_correlation_matrix
np.fill_diagonal(x, 0)

#%%  Generate multiple random
# store_flattened_random = list(np.tril(x).flatten())
# for i in range(1):
#     random_event_binned_data = generate_event_segmented(data=data.copy(), event_data=event_data)
#     random_nng = nng(random_event_binned_data)
#     x = random_nng.pearsons_correlation_matrix
#     np.fill_diagonal(x, 0)
#     store_flattened_random = store_flattened_random + list(np.tril(x).flatten())
# x = store_flattened_random
# x = np.array(x)

#%% Compute percentiles
Q1 = np.percentile(x, 25, interpolation='midpoint')
Q3 = np.percentile(x, 75, interpolation='midpoint')
print(Q3)
IQR = Q3 - Q1
outlier_threshold = Q3 + 1.5 * IQR
print(outlier_threshold)

ground_truth_nng_B = nng(data[:, 0:1800])
y = ground_truth_nng_B.pearsons_correlation_matrix
np.fill_diagonal(y, 0)

ground_truth_nng_A = nng(data[:, 1800:3600])
z = ground_truth_nng_A.pearsons_correlation_matrix
np.fill_diagonal(z, 0)

#%%
plt.ylim(0,700)
plt.hist(np.tril(x).flatten(), bins=50, color='grey', alpha=0.3)
plt.hist(np.tril(y).flatten(), bins=50, color='salmon', alpha=0.3)
plt.hist(np.tril(z).flatten(), bins=50, color='darkturquoise', alpha=0.3)
plt.axvline(x = outlier_threshold, color = 'red')
plt.legend(['threshold (Q3 + 1.5*IQR)', 'shuffled', 'ground truth',])
plt.xlabel("Pearson's r-value")
plt.ylabel("Frequency")
plt.title(file_str)
plt.savefig(EXPORT_PATH + f'{file_str}_{day}_threshold_histogram_binned_event.png', dpi=300)
plt.show()

from scipy import stats
random_vals = np.tril(x).flatten()
data_vals = np.tril(y).flatten()
print(f"KS-statistic: {stats.ks_2samp(random_vals, data_vals)}")

print(f"The threshold is: {outlier_threshold}")




#%% Plot event trace to visualize
plt.plot(random_event_binned_data[0, :], random_event_binned_data[1,:], color='grey', alpha=0.5)
plt.plot(data[0, 0:1800], data[1,0:1800], 'darkturquoise', alpha = 0.5)
plt.plot(data[0, 1800:3600], data[1,1800:3600], 'salmon', alpha = 0.5)