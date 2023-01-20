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
import os

from setup import FC_DATA_PATH
from dg_network_graph import DGNetworkGraph as nng

path_to_data = FC_DATA_PATH
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

RANDOM_DATA_PATH = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/analyses/benchmarking/random_networks/randomized_neural_data/'
EXPORT_PATH = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/analyses/benchmarking/random_networks/scratch-analysis/'

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

#
# def event_bins(data, events):
#     """
#     :param data:
#     :param events: single events timecourse
#     :return:
#     """
#     data = list(data)
#     build_binned_list = []
#     event_idx = list(np.nonzero(events)[0])
#     if not event_idx:
#         flat_random_binned_list = data
#     else:
#         if event_idx[-1] != len(data):
#             event_idx.append(len(data))
#         start_val = 0
#         for idx in event_idx:
#             # ## testing addition of breaking up no-event time periods
#             # if (idx - start_val > 30) and all(x == 0.005 for x in data[start_val:idx]): # if the next event is more than 30 timepoints from the last event, then break up the timecourse
#             #     while start_val < idx:
#             #         build_binned_list.append(data[start_val:start_val + 1]) # add to the list a series of chunks that are 5 timepoints in length
#             #         start_val += 1
#             #     if start_val == idx:
#             #         continue
#             #     else:
#             #         build_binned_list.pop()
#             #         start_val -= 1
#             #         build_binned_list.append(data[start_val:idx])
#             # ## testing
#             # else: # standard condition if the next event is less than 30 timepoints from the last
#             build_binned_list.append(data[start_val:idx])
#             start_val = idx
#         np.random.shuffle(build_binned_list)
#         flat_random_binned_list = [item for sublist in build_binned_list for item in sublist]
#     return flat_random_binned_list

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


def generate_event_segmented(data: list, event_data: list) -> np.ndarray:
    """
    data: list

    Parameter data should contain a list of np.ndarray objects.

    Return a numpy array or NWB file.
    """
    time = data[0, :].copy()

    # build binned dist
    flatten_array = time.copy()
    for row in range(np.shape(data[1:, :])[0]):
        binned_row = event_bins(data=data[row + 1, :], events = event_data[row + 1, :])
        flatten_array = np.vstack([flatten_array, binned_row])

    return flatten_array

def generate_randomized(data: list, bin_size: int) -> np.ndarray:
    """
    data: list

    Parameter data should contain a list of np.ndarray objects.

    Return a numpy array or NWB file.
    """
    time = data[0, :].copy()

    # build binned dist
    flatten_array = time.copy()
    for row in range(np.shape(data[2:, :])[0]):
        binned_row = bins(lst=data[row + 2, :], n=bin_size)
        flatten_array = np.vstack([flatten_array, binned_row])

    return flatten_array

#%% Demonstrate binned randomization example
file_str = '122-1'
ex_idx1 = 2
day = 'D1'
if day == 'D1':
    event_day = 'Day1'
else:
    event_day = 'Day9'
data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')
event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')
random_event_binned_data = generate_event_segmented(data=data.copy()[:, 0:1800], event_data=event_data)

# Plot the points of identified events
plt.figure(figsize=(15,5))
plt.plot(event_data[0,0:1800], event_data[ex_idx1,0:1800], '.')
# Check if context A only random networks are different than context B only, and how they compare to
plt.plot(random_event_binned_data[0, :], random_event_binned_data[2,:], color='grey', alpha=0.5)
plt.plot(data[0, 0:1800], data[ex_idx1,0:1800], 'darkturquoise', alpha = 0.5)
plt.title('Binned random data')
plt.show()


random_nng = nng(random_event_binned_data)
x = random_nng.pearsons_correlation_matrix
np.fill_diagonal(x, 0)

# Compute percentiles
Q1 = np.percentile(x, 10, method='inverted_cdf')
Q3 = np.percentile(x, 90, method='inverted_cdf')

IQR = Q3 - Q1
outlier_threshold = Q3 + 1.5 * IQR

ground_truth_nng_B = nng(data[:, 0:1800])
y = ground_truth_nng_B.pearsons_correlation_matrix
np.fill_diagonal(y, 0)

ground_truth_nng_A = nng(data[:, 1800:3600])
z = ground_truth_nng_A.pearsons_correlation_matrix
np.fill_diagonal(z, 0)

# Plot with threshold selected
plt.ylim(0,700)
plt.xlim(-0.2, 1.0)
plt.hist(np.tril(x).flatten(), bins=50, color='grey', alpha=0.3)
plt.hist(np.tril(y).flatten(), bins=50, color='darkturquoise', alpha=0.3)
# plt.hist(np.tril(z).flatten(), bins=50, color='salmon', alpha=0.3)
plt.axvline(x = outlier_threshold, color = 'red')
plt.legend(['threshold (Q3 + 1.5*IQR)', 'shuffled', 'ground truth',])
plt.xlabel("Pearson's r-value")
plt.ylabel("Frequency")
plt.title(file_str)
#plt.savefig(EXPORT_PATH + f'{file_str}_{day}_threshold_histogram_binned_event.png', dpi=300)
plt.show()

random_vals = np.tril(x).flatten()
data_vals = np.tril(y).flatten()
print(f"KS-statistic: {stats.ks_2samp(random_vals, data_vals)}")

print(f"The threshold is: {outlier_threshold}")

# Plot event trace to visualize
plt.figure(figsize=(15,5))
plt.plot(random_event_binned_data[0, :], random_event_binned_data[ex_idx1,:], color='grey', alpha=0.5)
plt.plot(data[0, 0:1800], data[ex_idx1,0:1800], 'darkturquoise', alpha = 0.5)
#plt.plot(data[0, 1800:3600], data[2,1800:3600], 'salmon', alpha = 0.5)
plt.show()

#%%
file_str = '2-2'
ex_idx1 = 2
day = 'D1'
if day == 'D1':
    event_day = 'Day1'
else:
    event_day = 'Day9'
data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')
event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')
random_event_binned_data = generate_event_segmented(data=data.copy()[:, 1800:], event_data=event_data)

# Plot the points of identified events
plt.figure(figsize=(15,5))
plt.plot(event_data[0,1800:], event_data[ex_idx1,1800:], '.')
# Check if context A only random networks are different than context B only, and how they compare to
plt.plot(random_event_binned_data[0, :], random_event_binned_data[2,:], color='grey', alpha=0.5)
plt.plot(data[0, 1800:], data[ex_idx1,1800:], 'salmon', alpha = 0.5)
plt.title('Binned random data')
plt.show()


random_nng = nng(random_event_binned_data)
x = random_nng.pearsons_correlation_matrix
x = np.nan_to_num(x)
np.fill_diagonal(x, 0)


# Compute percentiles
Q1 = np.percentile(np.tril(x).flatten(), 25, method='inverted_cdf')
Q3 = np.percentile(np.tril(x).flatten(), 75, method='inverted_cdf')

IQR = Q3 - Q1
outlier_threshold = Q3 + 1.5 * IQR

ground_truth_nng_B = nng(data[:,1800:])
y = ground_truth_nng_B.pearsons_correlation_matrix
np.fill_diagonal(y, 0)

ground_truth_nng_A = nng(data[:, 1800:])
z = ground_truth_nng_A.pearsons_correlation_matrix
np.fill_diagonal(z, 0)

# Plot with threshold selected
plt.ylim(0,700)
plt.xlim(-0.2, 1.0)
plt.hist(np.tril(x).flatten(), bins=50, color='grey', alpha=0.3)
#plt.hist(np.tril(y).flatten(), bins=50, color='dardkturquoise', alpha=0.3)
plt.hist(np.tril(z).flatten(), bins=50, color='salmon', alpha=0.3)
plt.axvline(x = outlier_threshold, color = 'red')
plt.legend(['threshold (Q3 + 1.5*IQR)', 'shuffled', 'ground truth',])
plt.xlabel("Pearson's r-value")
plt.ylabel("Frequency")
plt.title(file_str)
#plt.savefig(EXPORT_PATH + f'{file_str}_{day}_threshold_histogram_binned_event.png', dpi=300)
plt.show()

random_vals = np.tril(x).flatten()
data_vals = np.tril(z).flatten()
print(f"KS-statistic: {stats.ks_2samp(random_vals, data_vals)}")

print(f"The threshold is: {outlier_threshold}")

# Plot event trace to visualize
plt.figure(figsize=(15,5))
plt.plot(random_event_binned_data[0, :], random_event_binned_data[ex_idx1,:], color='grey', alpha=0.5)
plt.plot(data[0, 1800:], data[ex_idx1,1800:], 'salmon', alpha = 0.5)
plt.show()














#%% assemble list of ids
mouse_id_list = []
for file in os.listdir(path_to_data):
    if file.endswith('_smoothed_calcium_traces.csv'):
        print(file)
        mouse_id = file.replace('_smoothed_calcium_traces.csv', '')
        print(mouse_id)
        mouse_id_list.append(mouse_id)

#%% EVENT ANALYSIS: Only looking at Context B (first half of data)
# store: mouse_id, mean, median, max pearson, threshold event separated, ks-stat event separated, mean, median, max pearson, threshold bin-separated, ks-stat bin-separated
results = np.zeros((58, 13))

for count, mouse in enumerate(mouse_id_list):
    file_str = mouse[:len(mouse)-3]
    day = mouse[len(mouse)-2:]
    if day == 'D5' or day == 'D0':
        continue
    else:
        if day == 'D1':
            event_day = 'Day1'
        else:
            event_day = 'Day9'
        data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,0:1800]
        event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')
        random_event_binned_data = generate_event_segmented(data=data.copy(), event_data=event_data)

        random_nng = nng(random_event_binned_data)
        x = random_nng.pearsons_correlation_matrix
        np.fill_diagonal(x, 0)
        x = np.nan_to_num(x)

        # Store event trace results
        results[count, 0] = np.median(x)
        results[count, 1] = np.mean(x)
        results[count, 2] = np.max(x)

        # Compute percentiles
        Q1 = np.percentile(x, 25, method='inverted_cdf')
        Q3 = np.percentile(x, 75, method='inverted_cdf')
        IQR = Q3 - Q1
        outlier_threshold = Q3 + 1.5 * IQR

        ground_truth_nng_B = nng(data)
        y = ground_truth_nng_B.pearsons_correlation_matrix
        np.fill_diagonal(y, 0)
        y = np.nan_to_num(y)

        # Store event trace results
        results[count, 3] = np.median(y)
        results[count, 4] = np.mean(y)
        results[count, 5] = np.max(y)




        random_vals = np.tril(x).flatten()
        data_vals = np.tril(y).flatten()
        test_result = stats.ks_2samp(random_vals, data_vals)
        # print(f"Binsize = 1 KS-statistic: {test_result}")
        if test_result.pvalue > 0.05:
            print(f"{file_str} not significant")
        else:
            # Plot with threshold selected
            plt.ylim(0, 700)
            plt.hist(np.tril(x).flatten(), bins=50, color='grey', alpha=0.3)
            plt.hist(np.tril(y).flatten(), bins=50, color='darkturquoise', alpha=0.3)
            plt.axvline(x=outlier_threshold, color='red')
            plt.legend(['threshold', 'shuffled', 'ground truth'])
            plt.xlabel("Pearson's r-value")
            plt.ylabel("Frequency")
            plt.title(file_str + '_' + day + ': event-binned')
            plt.savefig(EXPORT_PATH + f'{file_str}_{day}_threshold_histogram_binned_event.png', dpi=300)
            plt.show()


        #print(f"The threshold is: {outlier_threshold}")

        # Store event trace results
        results[count, 6] = outlier_threshold
        results[count, 7] = stats.ks_2samp(random_vals, data_vals)[1]


#%%
import pandas as pd
pd.set_option("display.max.columns", None)

column_titles = ['rand mean', 'rand median', 'rand max', 'true mean', 'true median', 'true max', 'threshold', 'ks-statistic']
df = pd.DataFrame(results[:, 0:8], mouse_id_list, column_titles)
#df.insert(0, 'mouse-id', mouse_id_list)
print(df)

# drop all zero-valued rows
df = df.loc[~(df==0.000000).all(axis=1)]

# drop all NaN rows
df = df.dropna(axis=0)

#%%
threshold_analysis_df = df

#%%
average_threshold = threshold_analysis_df['threshold'].mean()
print(average_threshold)
#%% Store as hdf5
import numpy as np
import pandas as pd

save_path = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/analyses/benchmarking/random_networks/'
threshold_analysis_df.to_hdf(save_path + "threshold_analysis_df.h5", key = 'threshold_analysis_df', mode='w')
# Read saved HDF5
df = pd.read_hdf(save_path + "threshold_analysis_df.h5", 'threshold_analysis_df')



#%% Repeat
# Only looking at Context A (second half of data)
# store: mouse_id, mean, median, max pearson, mean, median, max pearson, threshold bin-separated, ks-stat bin-separated
results = np.zeros((58, 13))

for count, mouse in enumerate(mouse_id_list):
    file_str = mouse[:len(mouse)-3]
    day = mouse[len(mouse)-2:]
    if day == 'D5' or day == 'D0':
        continue
    else:
        if day == 'D1':
            event_day = 'Day1'
        else:
            event_day = 'Day9'
        data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')[:,1800:]
        event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')
        random_event_binned_data = generate_event_segmented(data=data.copy(), event_data=event_data)

        random_nng = nng(random_event_binned_data)
        x = random_nng.pearsons_correlation_matrix
        x = x/np.max(x)
        np.fill_diagonal(x, 0)
        x = np.nan_to_num(x)

        # Store event trace results
        results[count, 0] = np.median(x)
        results[count, 1] = np.mean(x)
        results[count, 2] = np.max(x)

        # Compute percentiles
        Q1 = np.percentile(x, 25, method='inverted_cdf')
        Q3 = np.percentile(x, 75, method='inverted_cdf')
        IQR = Q3 - Q1
        outlier_threshold = Q3 + 1.5 * IQR

        ground_truth_nng_A = nng(data)
        y = ground_truth_nng_A.pearsons_correlation_matrix
        y = y/np.max(y)
        np.fill_diagonal(y, 0)
        y = np.nan_to_num(y)

        # Store event trace results
        results[count, 3] = np.median(y)
        results[count, 4] = np.mean(y)
        results[count, 5] = np.max(y)

        random_vals = np.tril(x).flatten()
        data_vals = np.tril(y).flatten()
        test_result = stats.ks_2samp(random_vals, data_vals)
        # print(f"Binsize = 1 KS-statistic: {test_result}")
        if test_result.pvalue > 0.05:
            print(f"{file_str} {day} not significant")
            # Plot with threshold selected
            print(data)
            print(y)
        else:
            plt.ylim(0, 700)
            plt.xlim(-0.5, 1)
            plt.hist(np.tril(x).flatten(), bins=50, color='grey', alpha=0.3)
            plt.hist(np.tril(y).flatten(), bins=50, color='salmon', alpha=0.3)
            plt.axvline(x=outlier_threshold, color='red')
            plt.legend(['threshold', 'shuffled', 'ground truth'])
            plt.xlabel("Pearson's r-value")
            plt.ylabel("Frequency")
            plt.title(file_str + '_' + day)
            # plt.savefig(EXPORT_PATH + f'{file_str}_{day}_threshold_histogram_binned_event.png', dpi=300)
            plt.show()


        # Store event trace results
        results[count, 6] = outlier_threshold
        results[count, 7] = stats.ks_2samp(random_vals, data_vals)[1]


#%%
results = np.zeros((58, 13))

for count, mouse in enumerate(mouse_id_list):
    file_str = mouse[:len(mouse)-3]
    day = mouse[len(mouse)-2:]
    if day == 'D5' or day == 'D0':
        continue
    else:
        if day == 'D1':
            event_day = 'Day1'
        else:
            event_day = 'Day9'
        data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')
        event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')
        random_event_binned_data = generate_event_segmented(data=data.copy()[:, 1800:], event_data=event_data)


        random_nng = nng(random_event_binned_data)
        x = random_nng.pearsons_correlation_matrix
        x = np.nan_to_num(x)
        np.fill_diagonal(x, 0)


        # Compute percentiles
        Q1 = np.percentile(np.tril(x).flatten(), 25, method='inverted_cdf')
        Q3 = np.percentile(np.tril(x).flatten(), 75, method='inverted_cdf')

        IQR = Q3 - Q1
        outlier_threshold = Q3 + 1.5 * IQR

        ground_truth_nng_B = nng(data[:,1800:])
        y = ground_truth_nng_B.pearsons_correlation_matrix
        np.fill_diagonal(y, 0)

        ground_truth_nng_A = nng(data[:, 1800:])
        z = ground_truth_nng_A.pearsons_correlation_matrix
        np.fill_diagonal(z, 0)

        # Plot with threshold selected
        plt.ylim(0,700)
        plt.xlim(-0.2, 1.0)
        plt.hist(np.tril(x).flatten(), bins=50, color='grey', alpha=0.3)
        #plt.hist(np.tril(y).flatten(), bins=50, color='dardkturquoise', alpha=0.3)
        plt.hist(np.tril(z).flatten(), bins=50, color='salmon', alpha=0.3)
        plt.axvline(x = outlier_threshold, color = 'red')
        plt.legend(['threshold (Q3 + 1.5*IQR)', 'shuffled', 'ground truth',])
        plt.xlabel("Pearson's r-value")
        plt.ylabel("Frequency")
        plt.title(file_str)
        #plt.savefig(EXPORT_PATH + f'{file_str}_{day}_threshold_histogram_binned_event.png', dpi=300)
        plt.show()

        random_vals = np.tril(x).flatten()
        data_vals = np.tril(z).flatten()
        test_result = stats.ks_2samp(random_vals, data_vals)
        # print(f"Binsize = 1 KS-statistic: {test_result}")
        if test_result.pvalue > 0.05:
            print(f"{file_str} {day} not significant")
            # Plot with threshold selected
            print(data)
            print(y)
        else:
            plt.ylim(0, 700)
            plt.xlim(-0.5, 1)
            plt.hist(np.tril(x).flatten(), bins=50, color='grey', alpha=0.3)
            plt.hist(np.tril(y).flatten(), bins=50, color='salmon', alpha=0.3)
            plt.axvline(x=outlier_threshold, color='red')
            plt.legend(['threshold', 'shuffled', 'ground truth'])
            plt.xlabel("Pearson's r-value")
            plt.ylabel("Frequency")
            plt.title(file_str + '_' + day)
            # plt.savefig(EXPORT_PATH + f'{file_str}_{day}_threshold_histogram_binned_event.png', dpi=300)
            plt.show()

#%%
results = np.zeros((58, 13))

for count, mouse in enumerate(mouse_id_list):
    file_str = mouse[:len(mouse)-3]
    day = mouse[len(mouse)-2:]
    if day == 'D5' or day == 'D0':
        continue
    else:
        if day == 'D1':
            event_day = 'Day1'
        else:
            event_day = 'Day9'
        data = np.genfromtxt(path_to_data + f'/{file_str}_{day}_smoothed_calcium_traces.csv', delimiter=',')
        event_data = np.genfromtxt(path_to_data + f'/{file_str}_{event_day}_eventTrace.csv', delimiter=',')
        random_event_binned_data = generate_event_segmented(data=data.copy()[:, 0:1800], event_data=event_data)


        random_nng = nng(random_event_binned_data)
        x = random_nng.pearsons_correlation_matrix
        x = np.nan_to_num(x)
        np.fill_diagonal(x, 0)


        # Compute percentiles
        Q1 = np.percentile(np.tril(x).flatten(), 25, method='inverted_cdf')
        Q3 = np.percentile(np.tril(x).flatten(), 75, method='inverted_cdf')

        IQR = Q3 - Q1
        outlier_threshold = Q3 + 1.5 * IQR

        ground_truth_nng_B = nng(data[:,0:1800])
        y = ground_truth_nng_B.pearsons_correlation_matrix
        np.fill_diagonal(y, 0)

        ground_truth_nng_B = nng(data[:, 0:1800])
        z = ground_truth_nng_B.pearsons_correlation_matrix
        np.fill_diagonal(z, 0)

        # Plot with threshold selected
        plt.ylim(0,700)
        plt.xlim(-0.2, 1.0)
        plt.hist(np.tril(x).flatten(), bins=50, color='grey', alpha=0.3)
        #plt.hist(np.tril(y).flatten(), bins=50, color='dardkturquoise', alpha=0.3)
        plt.hist(np.tril(z).flatten(), bins=50, color='darkturquoise', alpha=0.3)
        plt.axvline(x = outlier_threshold, color = 'red')
        plt.legend(['threshold (Q3 + 1.5*IQR)', 'shuffled', 'ground truth',])
        plt.xlabel("Pearson's r-value")
        plt.ylabel("Frequency")
        plt.title(file_str)
        #plt.savefig(EXPORT_PATH + f'{file_str}_{day}_threshold_histogram_binned_event.png', dpi=300)
        plt.show()

        random_vals = np.tril(x).flatten()
        data_vals = np.tril(z).flatten()
        test_result = stats.ks_2samp(random_vals, data_vals)
        # print(f"Binsize = 1 KS-statistic: {test_result}")
        if test_result.pvalue > 0.05:
            print(f"{file_str} {day} not significant")
            # Plot with threshold selected
            print(data)
            print(y)
        else:
            plt.ylim(0, 700)
            plt.xlim(-0.5, 1)
            plt.hist(np.tril(x).flatten(), bins=50, color='grey', alpha=0.3)
            plt.hist(np.tril(y).flatten(), bins=50, color='darkturquoise', alpha=0.3)
            plt.axvline(x=outlier_threshold, color='red')
            plt.legend(['threshold', 'shuffled', 'ground truth'])
            plt.xlabel("Pearson's r-value")
            plt.ylabel("Frequency")
            plt.title(file_str + '_' + day)
            # plt.savefig(EXPORT_PATH + f'{file_str}_{day}_threshold_histogram_binned_event.png', dpi=300)
            plt.show()


# %%
pd.set_option("display.max.columns", None)

column_titles = ['rand mean', 'rand median', 'rand max', 'true mean', 'true median', 'true max', 'threshold',
                         'ks-statistic']
df = pd.DataFrame(results[:, 0:8], mouse_id_list, column_titles)
# df.insert(0, 'mouse-id', mouse_id_list)
print(df)

# drop all zero-valued rows
df = df.loc[~(df == 0.000000).all(axis=1)]

# drop all NaN rows
df = df.dropna(axis=0)

# %%
threshold_analysis_df = df

# %%
average_threshold = threshold_analysis_df['threshold'].mean()
print(average_threshold)

# %% Store as hdf5
import numpy as np
import pandas as pd

        save_path = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/analyses/benchmarking/random_networks/'
        threshold_analysis_df.to_hdf(save_path + "threshold_analysis_df.h5", key='threshold_analysis_df', mode='w')
        # Read saved HDF5
        df = pd.read_hdf(save_path + "threshold_analysis_df.h5", 'threshold_analysis_df')
