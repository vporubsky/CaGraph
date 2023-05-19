"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

Description: a preprocessing module to process the uploaded calcium imaging datasets before they are used in the
the CaGraph class to perform graph theory analysis.
"""
# Imports
from oasis.functions import deconvolve
from pynwb import NWBHDF5IO
import random
import numpy
import numpy as np
import matplotlib.pyplot as plt
import scipy
import warnings
import os


# Todo: check alternative options to report (with print statements currently)

#%% Utility functions
def _input_validator(data):
    """
    Validates the input dataset by checking that it is a numpy.ndarray or path to CSV or NWB file.

    :param data:
    :return:
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, str):
        if data.endswith('csv'):
            return np.genfromtxt(data, delimiter=",")
        elif data.endswith('nwb'):
            with NWBHDF5IO(data, 'r') as io:
                nwbfile_read = io.read()
                nwb_acquisition_key = list(nwbfile_read.acquisition.keys())[0]
                ca_from_nwb = nwbfile_read.acquisition[nwb_acquisition_key]
                neuron_dynamics = ca_from_nwb.data[:]
                time = ca_from_nwb.timestamps[:]
                return np.vstack((time, neuron_dynamics))
        else:
            raise TypeError('File path must have a .csv or .nwb file to load.')
    else:
        raise TypeError('Data must be passed as a str containing a .csv or .nwb file, or as numpy.ndarray.')

def _get_pearsons_correlation_matrix(data):
    """
    Returns the Pearson's correlation for all neuron pairs.

    :param data: numpy.ndarray
    :param time_points: tuple
    :return: numpy.ndarray
    """
    return np.nan_to_num(np.corrcoef(data, rowvar=True))

#%% ---------------- Clean data --------------------------------------
def deconvolve_dataset(data):
    """
    Uses OASIS algorithm implementation by J. Friedrich (https://github.com/j-friedrich/OASIS) to deconvolve the
    caclium imaging trace and infer neural activity.

    The inferred neural activity is converted to a binary array to construct the event data.

    :param data:
    :return:
    """
    data = _input_validator(data=data)
    decon_data = data[0, :]
    event_data = data[0, :]
    for neuron in range(1, data.shape[0]):
        # Perform OASIS deconvolution
        decon_trace, event_trace = deconvolve(data[neuron, :], penalty=0)[0:2]

        # Binarize event trace
        event_trace[event_trace < 1] = 0
        event_trace[event_trace >= 1] = 1

        # Stack trace onto datasets
        decon_data = np.vstack((decon_data, decon_trace))
        event_data = np.vstack((event_data, event_trace))
    return decon_data, event_data


#%% --------- Suitability for graph theory analysis -------------------------
# Todo: Update  threshold selection using bayesian inference or other metric
# Todo: update _event_bins() --> check that end threshold is necessary
def _event_bins(data_row, events):
    """
    Generates a shuffled row of calcium fluorescence data, breaking the timeseries using predetermined events.

    :param data_row: numpy.ndarray
    :param events: list
    :return: list
    """
    data = list(data_row)
    build_binned_list = []

    # Add all zero-valued points in the fluorescence data to the event trace
    zero_indices = np.where(np.array(data_row) < 0.001)  # 0.001 selected as threshold
    events = np.array(events)
    events[list(zero_indices)] = 1

    # Use event trace to split timeseries into relevant chunks to be shuffled
    events = list(events)
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
    data = _input_validator(data=data)
    if event_data is not None:
        event_data = _input_validator(data=event_data)
    else:
        _, event_data = deconvolve_dataset(data=data)
    time = data[0, :].copy()

    # build event-binned array
    flatten_array = time.copy()

    # iterate over the dataset and construct event-binned rows
    for row in range(np.shape(data[1:, :])[0]):
        binned_row = _event_bins(data_row=data[row + 1, :], events=event_data[row + 1, :])
        flatten_array = np.vstack([flatten_array, binned_row])
    return flatten_array


def generate_average_threshold(data, event_data=None, shuffle_iterations=100):
    """
    Performs multiple random shuffles to identify threshold values, and averages them.

    :param event_data:
    :param data: numpy.ndarray
    :param shuffle_iterations: int
    :return:
    """
    data = _input_validator(data)
    thresholds = []
    for i in range(shuffle_iterations):
        thresholds += [generate_threshold(data=data, event_data=event_data)]
    return np.mean(thresholds)

# Todo: remove this
def generate_threshold_distributions(data, event_data=None, report_threshold=False, report_test=False):
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
    data = _input_validator(data)
    if event_data is not None:
        shuffled_data = generate_event_shuffle(data=data, event_data=event_data)
    elif event_data is None:
        _, event_data = deconvolve_dataset(data=data)
        shuffled_data = generate_event_shuffle(data=data, event_data=event_data)
    x = _get_pearsons_correlation_matrix(data=shuffled_data)
    np.fill_diagonal(x, 0)

    y = _get_pearsons_correlation_matrix(data=data)
    np.fill_diagonal(y, 0)

    shuffled_vals = np.tril(x).flatten()
    data_vals = np.tril(y).flatten()
    return shuffled_vals, data_vals

# Todo: add checks on dataset --> if numpy, if csv
def generate_threshold(data, event_data=None, report_threshold=False, report_test=False):
    """
    Compares provided dataset and a shuffled dataset to propose a threshold to use to construct graph objects.

    Provides warning if the correlation distribution of the provided dataset is not statistically different from that of
    the shuffled dataset.

    :param report_threshold:
    :param data: numpy.ndarray
    :param shuffled_data: numpy.ndarray
    :param event_data:
    :param report_test: bool
    :return: float or dict
    """
    # Check that the input data is in the correct format and load dataset
    data = _input_validator(data=data)
    if event_data is not None:
        shuffled_data = generate_event_shuffle(data=data, event_data=event_data)
    elif event_data is None:
        _, event_data = deconvolve_dataset(data=data)
        shuffled_data = generate_event_shuffle(data=data, event_data=event_data)
    else:
        shuffled_data = generate_event_shuffle(data=data)
    x = _get_pearsons_correlation_matrix(data=shuffled_data)
    np.fill_diagonal(x, 0)

    # set threshold as the 99th percentile of the shuffle distribution
    threshold = np.percentile(x, 99, interpolation='midpoint')

    y = _get_pearsons_correlation_matrix(data=data)
    np.fill_diagonal(y, 0)

    shuffled_vals = np.tril(x).flatten()
    data_vals = np.tril(y).flatten()
    ks_statistic = scipy.stats.ks_2samp(shuffled_vals, data_vals)
    p_val = ks_statistic.pvalue
    if p_val < 0.05 and report_threshold:
        print(f"The threshold is: {threshold:.2f}")
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
        threshold_dict["threshold"] = threshold
        return threshold_dict
    return threshold


def plot_threshold(data, event_data=None, title=None, y_lim=None, show_plot=True, save_plot=False, save_path=None, dpi=300, format='png'):
    """
    Plots the correlation distributions of the dataset and the shuffled dataset, along with the identified threshold value.

    :param data: numpy.ndarray
    :param event_data: list
    :param show_plot: bool
    :return:
    """
    data = _input_validator(data=data)
    if event_data is not None:
        shuffled_data = generate_event_shuffle(data=data, event_data=event_data)
    elif event_data is None:
        _, event_data = deconvolve_dataset(data=data)
        shuffled_data = generate_event_shuffle(data=data, event_data=event_data)
    else:
        shuffled_data = generate_event_shuffle(data=data)
    x = _get_pearsons_correlation_matrix(data=shuffled_data)
    np.fill_diagonal(x, 0)

    # set threshold as the 99th percentile of the shuffle distribution
    threshold = np.percentile(x, 99, interpolation='midpoint')

    y = _get_pearsons_correlation_matrix(data=data)
    np.fill_diagonal(y, 0)

    # specify the bin width
    bin_width = 0.01

    # calculate the number of bins
    x_bins = int(np.ceil((x.max() - x.min()) / bin_width))
    y_bins = int(np.ceil((y.max() - y.min()) / bin_width))

    plt.xlim(-1.0, 1.0)
    if y_lim is not None:
        plt.ylim(0, y_lim)
    plt.hist(np.tril(x).flatten(), bins=x_bins, color='grey', alpha=0.3)
    plt.hist(np.tril(y).flatten(), bins=y_bins, color='blue', alpha=0.3)
    plt.axvline(x=threshold, color='red')
    plt.legend(['threshold', 'shuffled', 'ground truth', ])
    plt.xlabel("Pearson's r-value")
    plt.ylabel("Frequency")
    if title is not None:
        plt.title(title)
    if save_plot:
        if save_path is None:
            save_path = os.getcwd() + f'fig'
        plt.savefig(fname=save_path, dpi=dpi, format=format)
    if show_plot:
        plt.show()


def plot_shuffle_example(data, event_data=None, neuron_idx=None, show_plot=True, save_plot=False, save_path=None, format= 'png', dpi=300):
    """
    Plot shuffled distribution.

    :param data: numpy.ndarray
    :param event_data: list
    :param show_plot: bool
    :return:
    """
    data = _input_validator(data=data)
    if event_data is not None:
        shuffled_data = generate_event_shuffle(data=data, event_data=event_data)
    elif event_data is None:
        shuffled_data = generate_event_shuffle(data=data)
    if neuron_idx is None:
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
    if save_plot:
        if save_path is not None:
            save_path = save_path
        else:
            save_path = os.getcwd() + f'fig'
        plt.savefig(fname=save_path, dpi=dpi, format=format)
    if show_plot:
        plt.show()


# Todo: add function to plot event trace
def plot_event_trace():
    """

    :return:
    """
    return


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
    data = _input_validator(data=data)
    x = _get_pearsons_correlation_matrix(data=data[0])
    np.fill_diagonal(x, 0)

    y = _get_pearsons_correlation_matrix(data=data[1])
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
#     x = _get_pearsons_correlation_matrix(data=data1)
#     np.fill_diagonal(x, 0)
#
#     y = _get_pearsons_correlation_matrix(data=data2)
#     np.fill_diagonal(y, 0)
#
#     ks_statistic = scipy.stats.ks_2samp(x, y)
#     p_val = ks_statistic.pvalue
#     if p_val < sig_level:
#         print(f'Null hypothesis is rejected. KS P-value = {p_val:.3}')
#     else:
#         print(f'Null hypothesis is not rejected. KS P-value = {p_val:.3}')
