"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 

Description: 
"""
# Imports
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy
import warnings

#%% Key Todos
# Todo: check that automatic event detection is incorporated as default
# Todo: check alternative options to report (with print statements currently)
# Todo: add functionality to generate many random shuffles to compare to

#%%
def get_pearsons_correlation_matrix(data, time_points=None):
    """
    Returns the Pearson's correlation for all neuron pairs.

    :param data_matrix:
    :param time_points: tuple
    :return:
    """
    if time_points:
        data = data[:, time_points[0]:time_points[1]]
    return np.nan_to_num(np.corrcoef(data, rowvar=True))


# ---------------- Clean data --------------------------------------
# Todo: add smoothing algorithm used for CNMF for calcium imaging data
def smooth(data):
    """
    Smooth unprocessed data to remove noise.
    """
    smoothed_data = data
    return smoothed_data


# Todo: add auto-clean option for raw extracted calcium imaging time-series
def auto_preprocess(data):
    """

    """
    preprocessed_data = smooth(data)
    return preprocessed_data


# Todo: try to incorporate MCMC spike inference code for event detection, this is used in the CNMF
def get_events(data):
    """
    Generates event data using smoothed calcium imaging trace.

    :param data:
    :return:
    """
    event_data = data[0,:]
    for i in range(1, np.shape(data)[0]):
        event_row = get_row_event_data(row_data=data[i])
        event_data = np.vstack((event_data, event_row))
    return event_data

# Todo: clean up functionality - repurpose count_sign_switch code
def get_row_event_data(row_data):
    subtract = row_data[0:len(row_data) - 1] - row_data[1:]
    a = subtract
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)

    # get indices where sign change occurs in timeseries
    sign_idx = [i for i, x in enumerate(signchange) if x]

    # remove duplicate spikes
    sign_idx = np.array(sign_idx)
    duplicate_finder = sign_idx[1:]-sign_idx[0:len(sign_idx)-1]
    duplicates = [i for i, x in enumerate(duplicate_finder) if x == 1]
    removed_duplicates = list(sign_idx)
    for index in sorted(duplicates, reverse=True):
        del removed_duplicates[index]

    # build event row
    event_row = np.zeros(len(row_data))
    for index in removed_duplicates:
        event_row[index] = 1
    return event_row

# Todo: make private
def count_sign_switch(row_data):
    """
    Searches time-series data for points at which the time-series changes from increasing to
    decreasing or from decreasing to increasing.

    """
    subtract = row_data[0:len(row_data) - 1] - row_data[1:]
    a = subtract
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    return np.sum(signchange)

# Todo: confirm functionality is useful and check threshold value
def remove_low_activity(data, event_data, event_num_threshold=5):
    """
    Removes neurons with fewer than event_num_threshold events.

    Returns a new array of data without neurons that have low activity.
    """
    # apply activity treshold
    new_event_data = np.zeros((1, np.shape(event_data)[1]))
    new_data = np.zeros((1, np.shape(data)[1]))
    for row in range(np.shape(data)[0]):
        if count_sign_switch(row_data=data[row, :]) <= 5 and not row == 0:
            continue
        else:
            new_event_data = np.vstack((new_event_data, event_data[row, :]))
            new_data = np.vstack((new_data, data[row, :]))
    return new_data[1:, :], new_event_data[1:, :]

# Todo: determine if this is still required
def remove_quiescent(data, event_data, event_num_threshold=5):
    """
    data: numpy.ndarray
    event_bins: numpy.ndarray

    Removes inactive neurons from the dataset using event_data which the user must pass.
    """
    binarized_event_data = np.where(event_data > 0.0005, 1, 0)
    new_event_data = np.zeros((1, np.shape(event_data)[1]))
    new_data = np.zeros((1, np.shape(data)[1]))
    for row in range(np.shape(binarized_event_data)[0]):
        if np.sum(binarized_event_data[row, :]) <= event_num_threshold:
            continue
        else:
            new_event_data = np.vstack((new_event_data, event_data[row, :]))
            new_data = np.vstack((new_data, data[row, :]))
    return new_data[1:, :], new_event_data[1:, :]


# Suitability for graph theory analysis
def __bins(lst, n):
    """
    Yield successive n-sized chunks from lst.
    """
    lst = list(lst)
    build_binned_list = []
    for i in range(0, len(lst), n):
        build_binned_list.append(lst[i:i + n])
    return build_binned_list


def generate_noise_shuffle(data: list) -> np.ndarray:
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


def generate_binned_shuffle(data: list, bin_size: int) -> np.ndarray:
    """
    data: list

    Parameter data should contain a list of np.ndarray objects.

    Return a numpy array or NWB file.
    """
    time = data[0, :].copy()
    build_new_array = np.array(__bins(lst=data[1, :], n=bin_size))

    # build binned dist
    for row in range(np.shape(data[2:, :])[0]):
        binned_row = __bins(lst=data[row + 2, :], n=bin_size)
        build_new_array = np.vstack([build_new_array, binned_row])

    for row in range(np.shape(build_new_array)[0]):
        np.random.shuffle(build_new_array[row, :])

    flatten_array = time.copy()
    for row in range(np.shape(build_new_array)[0]):
        flat_row = [item for sublist in build_new_array[row, :] for item in sublist]
        flatten_array = np.vstack([flatten_array, flat_row])

    return flatten_array


def __event_bins(data, events):
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
    threshold = 0.01
    flat_random_binned_list = [0 if value < threshold else value for value in flat_random_binned_list]
    return flat_random_binned_list


def generate_event_shuffle(data: list, event_data: list) -> np.ndarray:
    """
    data: list

    Parameter data should contain a list of np.ndarray objects.

    Return a numpy array or NWB file.
    """
    time = data[0, :].copy()

    # build binned dist
    flatten_array = time.copy()
    for row in range(np.shape(data[1:, :])[0]):
        binned_row = __event_bins(data=data[row + 1, :], events=event_data[row + 1, :])
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
        binned_row = __bins(lst=data[row + 2, :], n=bin_size)
        flatten_array = np.vstack([flatten_array, binned_row])

    return flatten_array


def generate_population_event_shuffle(data: np.ndarray, event_data: np.ndarray) -> np.ndarray:
    """

    :param data:
    :param event_data:
    :return:
    """
    time = data[0, :].copy()

    # Build long list of all neurons time bins, then shuffle and flatten
    build_binned_list = []
    for row in range(1, np.shape(data)[0]):
        data_row = list(data[row, :])
        event_idx = list(np.nonzero(event_data[row, :])[0])
        if event_idx[-1] != len(data_row):
            event_idx.append(len(data_row))
        start_val = 0
        for idx in event_idx:
            build_binned_list.append(data[start_val:idx])
            start_val = idx
    flat_random_binned_list = [item for sublist in build_binned_list for item in sublist]
    np.random.shuffle(flat_random_binned_list)  # shuffles list of all neural data

    flat_random_binned_list = [item for sublist in flat_random_binned_list for item in sublist]

    # Rebuild a shuffled data matrix the size of original data matrix
    shuffled_data = time.copy()
    start_idx = 0
    end_idx = len(time)
    for row in range(np.shape(data[1:, :])[0]):
        binned_row = flat_random_binned_list[start_idx: end_idx]
        shuffled_data = np.vstack([shuffled_data, binned_row])
        start_idx = end_idx
        end_idx += len(time)

    return shuffled_data


def plot_shuffle_example(data, shuffled_data=None, event_data=None, show_plot=True):
    """

    :param shuffled_data:
    :param data:
    :param event_data:
    :param show_plot:
    :return:
    """
    if shuffled_data is None and event_data is not None:
        shuffled_data = generate_event_shuffle(data=data, event_data=event_data)
    elif shuffled_data is None and event_data is None:
        raise AttributeError
    neuron_idx = random.randint(1, np.shape(data)[0] - 1)
    plt.figure(figsize=(10, 5))
    plt.subplot(211)
    plt.plot(data[0, :], data[neuron_idx, :], c='blue', label='ground truth')
    plt.ylabel('ΔF/F')
    plt.legend()
    plt.subplot(212)
    plt.plot(shuffled_data[0, :], shuffled_data[neuron_idx, :], c='grey', label='shuffled')
    plt.ylabel('')
    plt.ylabel('ΔF/F')
    plt.xlabel('Time')
    plt.legend()
    if show_plot:
        plt.show()

def generate_threshold(data, shuffled_data=None, event_data=None, report_test=False):
    """
    Analyzes a shuffled dataset to propose a threshold to use to construct graph objects.

    :param data:
    :param shuffled_data:
    :param event_data:
    :return:
    """
    if shuffled_data is None and event_data is not None:
        shuffled_data = generate_event_shuffle(data=data, event_data=event_data)
    elif shuffled_data is None and event_data is None:
        event_data = get_events(data=data)
        shuffled_data = generate_event_shuffle(data=data, event_data=event_data)
    x = get_pearsons_correlation_matrix(data=shuffled_data)
    np.fill_diagonal(x, 0)
    Q1 = np.percentile(x, 25, interpolation='midpoint')
    Q3 = np.percentile(x, 75, interpolation='midpoint')

    IQR = Q3 - Q1
    outlier_threshold = round(Q3 + 1.5 * IQR,2)

    y = get_pearsons_correlation_matrix(data=data)
    np.fill_diagonal(y, 0)

    random_vals = np.tril(x).flatten()
    data_vals = np.tril(y).flatten()
    ks_statistic = scipy.stats.ks_2samp(random_vals, data_vals)
    p_val = ks_statistic.pvalue
    if report_test:
        print(f"KS-statistic: {ks_statistic.statistic}")
        print(f"P-val: {p_val}")
    if p_val < 0.05:
        print(f"The threshold is: {outlier_threshold:.2f}")
    else:
        warnings.warn('The KS-test performed on the shuffled and ground truth datasets show that the p-value is greater '
                      'than a 5% significance level. Confirm that correlations in dataset are differentiable from random correlations'
                      'before setting a threshold.')
    return outlier_threshold

def plot_threshold(data, shuffled_data=None, event_data=None, show_plot=True):
    """

    :param data:
    :param shuffled_data:
    :param event_data:
    :param show_plot:
    :return:
    """
    if shuffled_data is None and event_data is not None:
        shuffled_data = generate_event_shuffle(data=data, event_data=event_data)
    elif shuffled_data is None and event_data is None:
        event_data = get_events(data=data)
        shuffled_data = generate_event_shuffle(data=data, event_data=event_data)

    x = get_pearsons_correlation_matrix(data=shuffled_data)
    np.fill_diagonal(x, 0)
    Q1 = np.percentile(x, 25, interpolation='midpoint')
    Q3 = np.percentile(x, 75, interpolation='midpoint')

    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR

    y = get_pearsons_correlation_matrix(data=data)
    np.fill_diagonal(y, 0)

    plt.ylim(0, 100)
    plt.hist(np.tril(x).flatten(), bins=50, color='grey', alpha=0.3)
    plt.hist(np.tril(y).flatten(), bins=50, color='blue', alpha=0.3)
    plt.axvline(x=outlier_threshold, color='red')
    plt.legend(['threshold', 'shuffled', 'ground truth', ])
    plt.xlabel("Pearson's r-value")
    plt.ylabel("Frequency")
    if show_plot:
        plt.show()


# Todo: function to test sensitivity analysis
def sensitivity_analysis():
    return

# Todo: add function to plot event trace
def plot_event_trace():
    return

# Todo: update for formal inclusion
def plot_hist(data1, data2, colors, legend=None, show_plot=True):
    """

    :param data:
    :param shuffled_data:
    :param event_data:
    :param show_plot:
    :return:
    """
    x = get_pearsons_correlation_matrix(data=data1)
    np.fill_diagonal(x, 0)

    y = get_pearsons_correlation_matrix(data=data2)
    np.fill_diagonal(y, 0)

    x = np.tril(x).flatten()
    y = np.tril(y).flatten()
    # binwidth = 0.05
    # bins = range(min(x), max(x) + binwidth, binwidth)

    plt.hist(x, bins=50, color=colors[0], alpha=0.3)
    plt.hist(y, bins=50, color=colors[1], alpha=0.3)
    if legend is not None:
        plt.legend([legend[0], legend[1]])
    plt.xlabel("Pearson's r-value")
    plt.ylabel("Frequency")
    plt.ylim((0,100))
    if show_plot:
        plt.show()

# Todo: formally include
# def compute_ks(data1, data2, sig_level = 0.05):
#     """
#
#     :param data1:
#     :param data2:
#     :return:
#     """
#     x = get_pearsons_correlation_matrix(data=data1)
#     np.fill_diagonal(x, 0)
#
#     y = get_pearsons_correlation_matrix(data=data2)
#     np.fill_diagonal(y, 0)
#
#     ks_statistic = scipy.stats.ks_2samp(x, y)
#     p_val = ks_statistic.pvalue
#     if p_val < sig_level:
#         print(f'Null hypothesis is rejected. KS P-value = {p_val:.3}')
#     else:
#         print(f'Null hypothesis is not rejected. KS P-value = {p_val:.3}')

