"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

Description: a preprocessing module to process the uploaded calcium imaging datasets before they are used in the
the CaGraph class to perform graph theory analysis.
"""
# Imports
import random

import numpy
import numpy as np
import matplotlib.pyplot as plt
import scipy
import warnings
import os


# %% Key Todos
# Todo: check that automatic event detection is incorporated as default
# Todo: check alternative options to report (with print statements currently)
# Todo: add functionality to generate many random shuffles to compare to

# %%
def get_pearsons_correlation_matrix(data):
    """
    Returns the Pearson's correlation for all neuron pairs.

    :param data: numpy.ndarray
    :param time_points: tuple
    :return: numpy.ndarray
    """
    return np.nan_to_num(np.corrcoef(data, rowvar=True))


# ---------------- Clean data --------------------------------------
# Todo: add function with smoothing algorithm used for CNMF for calcium imaging data
def smooth(data):
    """
    Smooth unprocessed data to remove noise.

    :param data: numpy.ndarray
    :return: numpy.ndarray
    """
    smoothed_data = data
    return smoothed_data


# Todo: add auto-clean option for raw extracted calcium imaging time-series
def auto_preprocess(data):
    """

    :param data: numpy.ndarray
    :return: preprocessed_data: numpy.ndarray
    """
    preprocessed_data = smooth(data)
    return preprocessed_data


# Todo: try to incorporate MCMC spike inference code for event detection, this is used in the CNMF
def get_events(data):
    """
    Generates event data using smoothed calcium imaging trace.

    :param data: numpy.ndarray
    :return: numpy.ndarray
    """
    event_data = data[0, :]
    for i in range(1, np.shape(data)[0]):
        event_row = get_row_event_data(row_data=data[i])
        event_data = np.vstack((event_data, event_row))
    return event_data


# Todo: clean up functionality - repurpose count_sign_switch code
def get_row_event_data(row_data):
    """


    :param row_data: numpy.ndarray
    :return: numpy.ndarray
    """
    subtract = row_data[0:len(row_data) - 1] - row_data[1:]
    a = subtract
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)

    # get indices where sign change occurs in timeseries
    sign_idx = [i for i, x in enumerate(signchange) if x]

    # remove duplicate spikes
    sign_idx = np.array(sign_idx)
    duplicate_finder = sign_idx[1:] - sign_idx[0:len(sign_idx) - 1]
    duplicates = [i for i, x in enumerate(duplicate_finder) if x == 1]
    removed_duplicates = list(sign_idx)
    for index in sorted(duplicates, reverse=True):
        del removed_duplicates[index]

    # build event row
    event_row = np.zeros(len(row_data))
    for index in removed_duplicates:
        event_row[index] = 1
    return event_row


def _count_sign_switch(row_data):
    """
    Searches time-series data for points at which the time-series changes from increasing to
    decreasing or from decreasing to increasing.

    :param row_data: numpy.ndarray
    :return: numpy.ndarray
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

    :param data: numpy.ndarray
    :param event_data: numpy.ndarray
    :param event_num_threshold: int
    :return: numpy.ndarray
    """
    # apply activity treshold
    new_event_data = np.zeros((1, np.shape(event_data)[1]))
    new_data = np.zeros((1, np.shape(data)[1]))
    for row in range(np.shape(data)[0]):
        if _count_sign_switch(row_data=data[row, :]) <= 5 and not row == 0:
            continue
        else:
            new_event_data = np.vstack((new_event_data, event_data[row, :]))
            new_data = np.vstack((new_data, data[row, :]))
    return new_data[1:, :], new_event_data[1:, :]


# Todo: determine if this is still required
def remove_quiescent(data, event_data, event_num_threshold=5):
    """
    Removes inactive neurons from the dataset using event_data which the user must pass.

    :param data: numpy.ndarray
    :param event_data: numpy.ndarray
    :param event_num_threshold: int
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


# %% Suitability for graph theory analysis
def _bins(data_row, bin_size):
    """
    Return successive bin_size-sized chunks from data_row.

    :param data_row: numpy.ndarry or list
    :param bin_size: int
    :return: list
    """
    data_row = list(data_row)
    build_binned_list = []
    for i in range(0, len(data_row), bin_size):
        build_binned_list.append(data_row[i:i + bin_size])
    return build_binned_list


def generate_noise_shuffle(data: list) -> np.ndarray:
    """
    Shuffle every data point randomly within each neuron's calcium fluorescence timeseries.

    :param data: list
    :return: numpy.ndarray
    """
    time = data[0, :].copy()
    for row in range(np.shape(data)[0]):
        np.random.shuffle(data[row, :])
    data[0, :] = time.copy()  # reset time row to original sampling
    return data


def generate_binned_shuffle(data: list, bin_size: int) -> np.ndarray:
    """
    Shuffle bin_size-sized bins randomly within each neuron's calcium fluorescence timeseries.

    :param data: numpy.ndarray
    :param bin_size: int
    :return: numpy.ndarray
    """
    time = data[0, :].copy()
    build_new_array = np.array(_bins(data=data[1, :], bin_size=bin_size))
    # build binned dist
    for row in range(np.shape(data[2:, :])[0]):
        binned_row = _bins(data=data[row + 2, :], bin_size=bin_size)
        build_new_array = np.vstack([build_new_array, binned_row])
    for row in range(np.shape(build_new_array)[0]):
        np.random.shuffle(build_new_array[row, :])
    flatten_array = time.copy()
    for row in range(np.shape(build_new_array)[0]):
        flat_row = [item for sublist in build_new_array[row, :] for item in sublist]
        flatten_array = np.vstack([flatten_array, flat_row])
    return flatten_array


def _event_bins(data_row, events):
    """
    Generates a shuffled row of calcium fluorescence data, breaking the timeseries using predetermined events.

    :param data: numpy.ndarray
    :param events: list
    :return: list
    """
    data = list(data_row)
    build_binned_list = []
    event_idx = list(np.nonzero(events)[0])
    if event_idx[-1] != len(data):
        event_idx.append(len(data))
    start_val = 0
    for idx in event_idx:
        build_binned_list.append(data[start_val:idx])
        start_val = idx
    np.random.shuffle(build_binned_list)
    flat_shuffled_binned_list = [item for sublist in build_binned_list for item in sublist]
    threshold = 0.01
    flat_shuffled_binned_list = [0 if value < threshold else value for value in flat_shuffled_binned_list]
    return flat_shuffled_binned_list


def generate_event_shuffle(data: numpy.ndarray, event_data=None) -> np.ndarray:
    """
    Generates a shuffled dataset using events to break each neuron's calcium fluorescence timeseries.

    :param data: numpy.ndarray
    :param event_data: list
    :return numpy.ndarray
    """
    if event_data is not None:
        event_data = event_data
    else:
        event_data = get_events(data=data)
    time = data[0, :].copy()

    # build event-binned array
    flatten_array = time.copy()
    for row in range(np.shape(data[1:, :])[0]):
        binned_row = _event_bins(data_row=data[row + 1, :], events=event_data[row + 1, :])
        flatten_array = np.vstack([flatten_array, binned_row])
    return flatten_array

def generate_shuffled(data: numpy.ndarray, bin_size: int) -> np.ndarray:
    """
    Shuffle bin_size-sized bins randomly within each neuron's calcium fluorescence timeseries.

    :param data: numpy.ndarray
    :param bin_size: int
    :return: numpy.ndarray
    """
    time = data[0, :].copy()
    # build binned dist
    flatten_array = time.copy()
    for row in range(np.shape(data[2:, :])[0]):
        binned_row = _bins(data_row=data[row + 2, :], bin_size=bin_size)
        flatten_array = np.vstack([flatten_array, binned_row])
    return flatten_array

def generate_population_event_shuffle(data: np.ndarray, event_data: np.ndarray) -> np.ndarray:
    """
    Shuffle event bins randomly across all neuron's calcium fluorescence timeseries.

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
    flat_shuffled_binned_list = [item for sublist in build_binned_list for item in sublist]
    np.random.shuffle(flat_shuffled_binned_list)  # shuffles list of all neural data

    flat_shuffled_binned_list = [item for sublist in flat_shuffled_binned_list for item in sublist]

    # Rebuild a shuffled data matrix the size of original data matrix
    shuffled_data = time.copy()
    start_idx = 0
    end_idx = len(time)
    for row in range(np.shape(data[1:, :])[0]):
        binned_row = flat_shuffled_binned_list[start_idx: end_idx]
        shuffled_data = np.vstack([shuffled_data, binned_row])
        start_idx = end_idx
        end_idx += len(time)

    return shuffled_data


def plot_shuffle_example(data, shuffled_data=None, event_data=None, show_plot=True):
    """
    Plot shuffled distribution.

    :param data: numpy.ndarray
    :param shuffled_data: numpy.ndarray
    :param event_data: list
    :param show_plot: bool
    :return:
    """
    if shuffled_data is None and event_data is not None:
        shuffled_data = generate_event_shuffle(data=data, event_data=event_data)
    elif shuffled_data is None and event_data is None:
        event_data = get_events(data=data)
        shuffled_data = generate_event_shuffle(data=data, event_data=event_data)
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

def generate_average_threshold(data, shuffle_iterations=100):
    """
    Performs multiple random shuffles to identify threshold values, and averages them.

    :param data: numpy.ndarray
    :param shuffle_iterations: int
    :return:
    """
    thresholds = []
    for i in range(shuffle_iterations):
        thresholds += [generate_threshold(data=data)]
    return np.mean(thresholds)

# Todo: add checks on dataset --> if numpy, if csv
def generate_threshold(data, shuffled_data=None, event_data=None, report_threshold=False, report_test=False):
    """
    Compares provided dataset and a shuffled dataset to propose a threshold to use to construct graph objects.

    Provides warning if the correlation distribution of the provided dataset is not statistically different from that of
    the shuffled dataset.

    :param data: numpy.ndarray
    :param shuffled_data: numpy.ndarray
    :param event_data:
    :param report_test: bool
    :return: float or dict
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
    outlier_threshold = round(Q3 + 1.5 * IQR, 2)

    y = get_pearsons_correlation_matrix(data=data)
    np.fill_diagonal(y, 0)

    shuffled_vals = np.tril(x).flatten()
    data_vals = np.tril(y).flatten()
    ks_statistic = scipy.stats.ks_2samp(shuffled_vals, data_vals)
    p_val = ks_statistic.pvalue
    if p_val < 0.05 and report_threshold:
        print(f"The threshold is: {outlier_threshold:.2f}")
    elif report_threshold:
        warnings.warn(
            'The KS-test performed on the shuffled and ground truth datasets show that the p-value is greater '
            'than a 5% significance level. Confirm that correlations in dataset are differentiable from shuffled correlations '
            'before setting a threshold.')
    if report_test:
        print(f"KS-statistic: {ks_statistic.statistic}")
        print(f"P-val: {p_val}")
        threshold_dict = {"KS-statistic": ks_statistic.statistic}
        threshold_dict["P-val"] = p_val
        threshold_dict["threshold"] = outlier_threshold
        return threshold_dict
    return outlier_threshold

def plot_threshold(data, shuffled_data=None, event_data=None, y_lim=None, show_plot=True):
    """
    Plots the correlation distributions of the dataset and the shuffled dataset, along with the identified threshold value.

    :param data: numpy.ndarray
    :param shuffled_data: numpy.ndarray
    :param event_data: list
    :param show_plot: bool
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

    # specify the bin width
    bin_width = 0.01

    # calculate the number of bins
    x_bins = int(np.ceil((x.max() - x.min()) / bin_width))
    y_bins = int(np.ceil((y.max() - y.min()) / bin_width))
    plt.xlim(-0.3, 1.0)
    if y_lim is not None:
        plt.ylim(0, y_lim)
    plt.hist(np.tril(x).flatten(), bins=x_bins, color='grey', alpha=0.3)
    plt.hist(np.tril(y).flatten(), bins=y_bins, color='blue', alpha=0.3)
    plt.axvline(x=outlier_threshold, color='red')
    plt.legend(['threshold', 'shuffled', 'ground truth', ])
    plt.xlabel("Pearson's r-value")
    plt.ylabel("Frequency")
    if show_plot:
        plt.show()


# Todo: function to test sensitivity analysis
def sensitivity_analysis():
    """

    :return:
    """
    return


# Todo: add function to plot event trace
def plot_event_trace():
    """

    :return:
    """
    return

# Todo: move to cagraph class/ merge with visualization version
def plot_histogram(data, colors, legend=None, title=None, y_label=None, x_label=None, show_plot=True, save_plot=False,
                   save_path=None, dpi=300, format='png'):
    """
    Plot histograms of the provided datasets.

    :param data: list
    :param colors: list
    :param legend: list
    :param title:
    :param y_label:
    :param x_label:
    :param show_plot:
    :param save_plot:
    :param save_path:
    :param dpi:
    :param format:
    :return:
    """
    # specify the bin width
    bin_width = 0.01

    for dataset in data:
        # calculate the number of bins
        dataset_bins = int(np.ceil((dataset.max() - dataset.min()) / bin_width))

        # plot histogram
        plt.hist(dataset, bins=dataset_bins, color=colors[0], alpha=0.3)

    if legend is not None:
        plt.legend(legend)
    if title is not None:
        plt.title(title)
    if y_label is not None:
        plt.ylabel(y_label)
    if x_label is not None:
        plt.xlabel(x_label)
    else:
        plt.ylabel("Frequency")
    if show_plot:
        plt.show()
    if save_plot:
        if save_path is None:
            save_path = os.getcwd() + f'fig'
        plt.savefig(fname=save_path, dpi=dpi, format=format)


def plot_correlation_hist(data, colors, legend=None, title=None, y_label=None, x_label=None, show_plot=True,
                          save_plot=False, save_path=None, dpi=300, format='png'):
    """
    Plot histograms of the Pearson's correlation coefficient distributions for the provided datasets.

    :param data: list
    :param colors: list
    :param legend: list
    :param title:
    :param y_label:
    :param x_label:
    :param show_plot:
    :param save_plot:
    :param save_path:
    :param dpi:
    :param format:
    :return:
    """
    x = get_pearsons_correlation_matrix(data=data[0])
    np.fill_diagonal(x, 0)

    y = get_pearsons_correlation_matrix(data=data[1])
    np.fill_diagonal(y, 0)

    x = np.tril(x).flatten()
    y = np.tril(y).flatten()

    # specify the bin width
    bin_width = 0.01

    # calculate the number of bins
    x_bins = int(np.ceil((x.max() - x.min()) / bin_width))
    y_bins = int(np.ceil((y.max() - y.min()) / bin_width))

    # plot histograms
    plt.hist(x, bins=x_bins, color=colors[0], alpha=0.3)
    plt.hist(y, bins=y_bins, color=colors[1], alpha=0.3)
    if legend is not None:
        plt.legend(legend)
    if title is not None:
        plt.title(title)
    if y_label is not None:
        plt.ylabel(y_label)
    if x_label is not None:
        plt.xlabel(x_label)
    else:
        plt.ylabel("Frequency")
    if show_plot:
        plt.show()
    if save_plot:
        if save_path is None:
            save_path = os.getcwd() + f'fig'
        plt.savefig(fname=save_path, dpi=dpi, format=format)

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




