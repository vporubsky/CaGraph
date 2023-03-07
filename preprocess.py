"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 

Description: 
"""
# Imports
from cagraph import CaGraph
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy

# ---------------- Clean data --------------------------------------
# Todo: add smoothing algorithm for calcium imaging data
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


# Todo: write event_detection code
def event_detection(data):
    """

    :param data:
    :return:
    """
    event_data = data
    return event_data


# Todo: create function to generate event_data
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

# Todo: make private
def __count_sign_switch(row_data):
    """
    Searches time-series data for points at which the time-series changes from increasing to
    decreasing or from decreasing to increasing.

    """
    subtract = row_data[0:len(row_data) - 1] - row_data[1:]
    a = subtract
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    return np.sum(signchange)


def remove_low_activity(data, event_data, event_num_threshold=5):
    """
    Removes neurons with fewer than event_num_threshold events.

    Returns a new array of data without neurons that have low activity.
    """
    # apply activity treshold
    new_event_data = np.zeros((1, np.shape(event_data)[1]))
    new_data = np.zeros((1, np.shape(data)[1]))
    for row in range(np.shape(data)[0]):
        if __count_sign_switch(row_data=data[row, :]) <= 5 and not row == 0:
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


def generate_threshold(data, shuffled_data=None, event_data=None):
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
        raise AttributeError
    random_cg = CaGraph(shuffled_data)
    x = random_cg.pearsons_correlation_matrix
    np.fill_diagonal(x, 0)
    Q1 = np.percentile(x, 25, interpolation='midpoint')
    Q3 = np.percentile(x, 75, interpolation='midpoint')

    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR

    ground_truth_cg = CaGraph(data)
    y = ground_truth_cg.pearsons_correlation_matrix
    np.fill_diagonal(y, 0)

    random_vals = np.tril(x).flatten()
    data_vals = np.tril(y).flatten()
    print(f"KS-statistic: {scipy.stats.ks_2samp(random_vals, data_vals)}")
    print(f"The threshold is: {outlier_threshold}")
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
        raise AttributeError('Must provide either a pre-processed shuffled dataset or event data to perform the shuffle.')
    random_cg = CaGraph(shuffled_data)
    x = random_cg.pearsons_correlation_matrix
    np.fill_diagonal(x, 0)
    Q1 = np.percentile(x, 25, interpolation='midpoint')
    Q3 = np.percentile(x, 75, interpolation='midpoint')

    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR

    ground_truth_cg = CaGraph(data)
    y = ground_truth_cg.pearsons_correlation_matrix
    np.fill_diagonal(y, 0)

    plt.ylim(0, 100)
    plt.hist(np.tril(x).flatten(), bins=50, color='grey', alpha=0.3)
    plt.hist(np.tril(y).flatten(), bins=50, color='blue', alpha=0.3)
    plt.axvline(x=outlier_threshold, color='red')
    plt.legend(['threshold (Q3 + 1.5*IQR)', 'shuffled', 'ground truth', ])
    plt.xlabel("Pearson's r-value")
    plt.ylabel("Frequency")
    if show_plot:
        plt.show()


# Todo: function to test sensitivity analysis
def sensitivity_analysis():
    return
