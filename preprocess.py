"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

Description: a preprocessing module to process the uploaded calcium imaging datasets before they are used in the
CaGraph class to perform graph theory analysis.
"""
# Imports
from oasis.functions import deconvolve
from pynwb import NWBHDF5IO
import random
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import scipy
import warnings
import os


# %% Utility functions
def _input_validator(data):
    """
    Validate and load input data from either a numpy array, CSV file, or NWB file.

    Parameters
    ----------
    data : str or np.ndarray
        Path to a .csv or .nwb file, or an in-memory NumPy array.

    Returns
    -------
    np.ndarray
        Data loaded as a NumPy array where rows represent neurons and columns are timepoints.

    Raises
    ------
    TypeError
        If data is not a supported format.
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

def get_correlation_matrix(data=None, method="pearson", crosscorr_max_lag=10, partial_alpha=1.0) -> np.ndarray:
    """
    Compute a correlation matrix for neural data using the specified method.

    Parameters
    ----------
    data : np.ndarray
        A 2D array with shape (neurons × timepoints).
    method : str
        Correlation type: 'pearson', 'spearman', 'crosscorr', or 'partial'.
    crosscorr_max_lag : int, optional
        Maximum time lag for cross-correlation.
    partial_alpha : float, optional
        Ridge regularization strength for partial correlation.

    Returns
    -------
    np.ndarray
        Square correlation matrix of shape (neurons × neurons).

    Raises
    ------
    ValueError
        If an unsupported method is specified.
    """
    n = data.shape[0]
    corr_matrix = np.ones((n, n))
    if method == "pearson":
        return np.nan_to_num(np.corrcoef(data, rowvar=True))
    elif method == "spearman":
        return np.nan_to_num(stats.spearmanr(data.T).correlation)
    elif method == "crosscorr":
        for i in range(n):
            for j in range(i + 1, n):
                x, y = data[i], data[j]
                corr = max(
                    [np.corrcoef(np.roll(x, lag), y)[0, 1] for lag in range(-crosscorr_max_lag, crosscorr_max_lag + 1)])
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        return corr_matrix
    elif method == "partial":
        for i in range(n):
            for j in range(i + 1, n):
                idx = [k for k in range(n) if k != i and k != j]
                Z = data[idx].T
                model_i = Ridge(alpha=partial_alpha).fit(Z, data[i])
                model_j = Ridge(alpha=partial_alpha).fit(Z, data[j])
                res_i = data[i] - model_i.predict(Z)
                res_j = data[j] - model_j.predict(Z)
                corr = np.corrcoef(res_i, res_j)[0, 1]
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
        return corr_matrix
    else:
        raise ValueError(f"Unsupported method: {method}")


# %% ---------------- Clean data --------------------------------------
def deconvolve_dataset(data, sn=0.2):
    """
    Uses OASIS algorithm implementation by J. Friedrich (https://github.com/j-friedrich/OASIS) to deconvolve the
    calcium imaging trace and infer neural activity.

    Parameters
    ----------
    data : str or np.ndarray
        Input data with time in the first row and neural signals in subsequent rows.
    sn : float, default=0.2
        Noise level for OASIS deconvolution.

    Returns
    -------
    tuple
        decon_data : np.ndarray
            Continuous deconvolved traces.
        event_data : np.ndarray
            Binarized event detections.
    """
    data = _input_validator(data=data)
    decon_data = data[0, :]
    event_data = data[0, :]
    for neuron in range(1, data.shape[0]):
        # Perform OASIS deconvolution
        decon_trace, event_trace = deconvolve(data[neuron, :], penalty=0, sn=sn)[0:2]

        # Binarize event trace
        event_trace[event_trace < 1] = 0
        event_trace[event_trace >= 1] = 1

        # Stack trace onto datasets
        decon_data = np.vstack((decon_data, decon_trace))
        event_data = np.vstack((event_data, event_trace))
    return decon_data, event_data


# %% --------- Suitability for graph theory analysis -------------------------
def _event_bins(data_row, events):
    """
    Segment a single neuron's timeseries based on event positions and shuffle the segments.

    Parameters
    ----------
    data_row : np.ndarray
        Fluorescence data for a single neuron.
    events : np.ndarray
        Binary vector indicating event positions.

    Returns
    -------
    list
        Shuffled list of time segments preserving event-based structure.
    """
    data = list(data_row)
    build_binned_list = []

    # Add all zero-valued points in the fluorescence data to the event trace
    zero_indices = np.where(np.array(data_row) < 0.001)  # 0.001 selected as threshold
    events = np.array(events)
    events[tuple(list(zero_indices))] = 1

    # Use event trace to split timeseries into relevant chunks to be shuffled
    events = list(events)
    event_idx = list(np.nonzero(events)[0])
    if len(event_idx) == 0:
        event_idx = [len(data)]
    if event_idx[-1] != len(data):
        event_idx.append(len(data))
    start_val = 0
    for idx in event_idx:
        build_binned_list.append(data[start_val:idx])
        start_val = idx
    np.random.shuffle(build_binned_list)
    flat_shuffled_binned_list = [item for sublist in build_binned_list for item in sublist]
    return flat_shuffled_binned_list


def generate_event_shuffle(data: np.ndarray, event_data=None) -> np.ndarray:
    """
    Generate a shuffled dataset using event-based segmentation for all neurons.

    Parameters
    ----------
    data : np.ndarray
        Original calcium dataset.
    event_data : np.ndarray, optional
        Binary event data per neuron.

    Returns
    -------
    np.ndarray
        Shuffled version of the dataset, aligned by time and structure.
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


def generate_correlation_distributions(data, shuffled_data=None, event_data=None):
    """
    Compute correlation coefficient distributions for original and shuffled datasets.

    Parameters
    ----------
    data : np.ndarray
        Original neural data matrix.
    shuffled_data : np.ndarray, optional
        Precomputed shuffled version of data.
    event_data : np.ndarray, optional
        Event data used to generate shuffle if not supplied.

    Returns
    -------
    tuple
        shuffled_vals : np.ndarray
            Correlation values from shuffled dataset (upper triangle).
        data_vals : np.ndarray
            Correlation values from real dataset (upper triangle).
    """
    data = _input_validator(data)
    if shuffled_data is None:
        if event_data is not None:
            shuffled_data = generate_event_shuffle(data=data, event_data=event_data)
        else:
            _, event_data = deconvolve_dataset(data=data)
            shuffled_data = generate_event_shuffle(data=data, event_data=event_data)
    x = get_correlation_matrix(data=shuffled_data, method="pearson")
    np.fill_diagonal(x, 0)

    y = get_correlation_matrix(data=data, method="pearson")
    np.fill_diagonal(y, 0)

    shuffled_vals = x[np.triu_indices_from(x, k=1)]
    data_vals = y[np.triu_indices_from(y, k=1)]
    return shuffled_vals, data_vals


def generate_average_threshold(data, event_data=None, correlation_method='pearson', shuffle_iterations=100, **correlation_kwargs):
    """
    Compute average correlation threshold by bootstrapping shuffled distributions.

    Parameters
    ----------
    data : np.ndarray
        Original calcium data.
    event_data : np.ndarray, optional
        Precomputed event data for shuffling.
    shuffle_iterations : int, default=100
        Number of shuffles to perform.

    Returns
    -------
    float
        Mean 99th-percentile threshold across iterations.
    """
    data = _input_validator(data)
    if event_data is None:
        _, event_data = deconvolve_dataset(data=data)
    thresholds = []
    for i in range(shuffle_iterations):
        thresholds += [generate_threshold(data=data, event_data=event_data, correlation_method=correlation_method, **correlation_kwargs)]
    return np.mean(thresholds)


def generate_threshold(data, shuffled_data=None, event_data=None, correlation_method='pearson', report_threshold=False, report_test=False,
                       return_test=False, **correlation_kwargs):
    """
    Estimate a correlation threshold by comparing original vs shuffled datasets.

    Parameters
    ----------
    data : np.ndarray
        Calcium imaging data.
    shuffled_data : np.ndarray, optional
        Precomputed shuffled dataset.
    event_data : np.ndarray, optional
        Event data for generating shuffle if not provided.
    correlation_method : str, default='pearson'
        Correlation method to use.
    report_threshold : bool, default=False
        Whether to print the threshold.
    report_test : bool, default=False
        Whether to print the KS-test results.
    return_test : bool, default=False
        If True, return KS statistics and threshold as a dict.
    **correlation_kwargs : dict
        Extra keyword arguments for correlation computation.

    Returns
    -------
    float or dict
        Threshold value or dict with KS-test results and threshold.
    """
    # Check that the input data is in the correct format and load dataset
    data = _input_validator(data=data)
    if shuffled_data is None:
        if event_data is not None:
            shuffled_data = generate_event_shuffle(data=data, event_data=event_data)
        elif event_data is None:
            _, event_data = deconvolve_dataset(data=data)
            shuffled_data = generate_event_shuffle(data=data, event_data=event_data)
        else:
            shuffled_data = generate_event_shuffle(data=data)
    data_correlation = get_correlation_matrix(data=data, method=correlation_method, **correlation_kwargs)
    np.fill_diagonal(data_correlation, 0)

    shuffle_correlation = get_correlation_matrix(data=shuffled_data, method=correlation_method, **correlation_kwargs)
    np.fill_diagonal(shuffle_correlation, 0)

    # set threshold as the 99th percentile of the shuffle distribution
    shuffled_vals = shuffle_correlation[np.triu_indices_from(shuffle_correlation, k=1)]
    data_vals = data_correlation[np.triu_indices_from(data_correlation, k=1)]

    threshold = np.percentile(shuffled_vals, 99, method='midpoint')

    shuffled_correlation = np.tril(shuffle_correlation).flatten()
    data_correlation = np.tril(data_correlation).flatten()

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
    if return_test:
        threshold_dict = {"KS-statistic": ks_statistic.statistic, "P-val": p_val, "threshold": threshold}
        return threshold_dict
    else:
        return threshold


def plot_threshold(data, shuffled_data=None, event_data=None, correlation_method='pearson', data_id=None,
                   data_color='blue', shuffle_color='grey', threshold_color='red', alpha=0.3,
                   title=None, x_label='Correlation metric', y_label='Frequency',
                   xlim=None, ylim=None, show_plot=True, save_plot=False, save_path=None, dpi=300,
                   save_format='png', plotting_kwargs=None, correlation_kwargs=None):
    """
    Plot the distribution of correlation coefficients for a calcium imaging dataset and its event-shuffled counterpart,
    along with a threshold based on the 99th percentile of the shuffled distribution.

    Parameters:
    ----------
    data : np.ndarray
        Calcium imaging dataset with shape (neurons × timepoints), or (time + neurons × timepoints).
        The first row should be time if present.

    shuffled_data : np.ndarray, optional
        Precomputed shuffled dataset. If None, one will be generated using `event_data`.

    event_data : np.ndarray, optional
        Event-triggered binary array used for generating shuffled data. If not provided,
        it will be computed using OASIS deconvolution.

    correlation_method : str, default='pearson'
        Method for computing the correlation matrix. Supported methods: 'pearson', 'spearman',
        'crosscorr', 'partial'.

    data_id : str, optional
        Identifier used in plot legends.

    data_color : str, default='blue'
        Color used to plot the histogram of the original dataset's correlations.

    shuffle_color : str, default='grey'
        Color used to plot the histogram of the shuffled dataset's correlations.

    threshold_color : str, default='red'
        Color for the vertical threshold line.

    alpha : float, default=0.3
        Transparency of the histogram bars.

    title : str, optional
        Title for the plot.

    x_label : str, default='Correlation metric'
        Label for the x-axis.

    y_label : str, default='Frequency'
        Label for the y-axis.

    xlim : tuple, optional
        x-axis limits.

    ylim : tuple, optional
        y-axis limits.

    show_plot : bool, default=True
        If True, displays the plot.

    save_plot : bool, default=False
        If True, saves the plot to file.

    save_path : str, optional
        Path to save the figure (including filename). If None, saves to the current directory.

    dpi : int, default=300
        Resolution of saved plot in dots-per-inch.

    save_format : str, default='png'
        Format to use when saving the plot.

    plotting_kwargs : dict, optional
        Additional keyword arguments passed to `matplotlib.pyplot.hist`.

    correlation_kwargs : dict, optional
        Additional keyword arguments passed to the correlation matrix computation
        (e.g., `crosscorr_max_lag=10`, `partial_alpha=1.0`).

    Returns:
    -------
    None
    """
    data = _input_validator(data=data)

    # Safely unpack keyword arguments
    plotting_kwargs = plotting_kwargs or {}
    correlation_kwargs = correlation_kwargs or {}

    if shuffled_data is None:
        if event_data is not None:
            shuffled_data = generate_event_shuffle(data=data, event_data=event_data)
        elif event_data is None:
            _, event_data = deconvolve_dataset(data=data)
            shuffled_data = generate_event_shuffle(data=data, event_data=event_data)
        else:
            shuffled_data = generate_event_shuffle(data=data)

    data_correlation = get_correlation_matrix(data=data, method=correlation_method, **correlation_kwargs)
    np.fill_diagonal(data_correlation, 0)

    shuffle_correlation = get_correlation_matrix(data=shuffled_data, method=correlation_method, **correlation_kwargs)
    np.fill_diagonal(shuffle_correlation, 0)

    # set threshold as the 99th percentile of the shuffle distribution
    shuffle_vals = shuffle_correlation[np.triu_indices_from(shuffle_correlation, k=1)]
    data_vals = data_correlation[np.triu_indices_from(data_correlation, k=1)]
    threshold = np.percentile(shuffle_vals, 99, method='midpoint')

    # calculate the number of bins
    bin_width = 0.01
    x_bins = int(np.ceil((shuffle_vals.max() - shuffle_vals.min()) / bin_width))
    y_bins = int(np.ceil((data_vals.max() - data_vals.min()) / bin_width))

    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    else:
        plt.xlim(-1.0, 1.0)

    # Plot histograms of shuffle, data, and threshold
    plt.hist(shuffle_vals, bins=x_bins, color=shuffle_color, alpha=alpha, **plotting_kwargs)
    plt.hist(data_vals, bins=y_bins, color=data_color, alpha=alpha, **plotting_kwargs)

    plt.axvline(x=threshold, color=threshold_color)

    # Specify plot details
    if data_id is not None:
        plt.legend(['threshold', f'shuffled {data_id}', f'{data_id}'], loc='upper left')
    elif data_id is None:
        plt.legend(['threshold', 'shuffled', 'ground truth'], loc='upper left')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title is not None:
        plt.title(title)
    if save_plot:
        if save_path is None:
            save_path = os.getcwd() + f'fig'
        elif not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(fname=save_path, bbox_inches='tight', dpi=dpi, format=save_format)
    if show_plot:
        plt.show()


# todo: reevaluate plotting kwargs --> duplicate may cause collisions
def plot_shuffled_neuron(data, shuffled_data=None, event_data=None, data_color='blue', shuffle_color='grey', neuron_index=None,
                         show_plot=True, save_plot=False, save_path=None,
                         save_format='png', dpi=300, **kwargs):
    """
    Plot a comparison between original and event-shuffled neural traces for a single neuron.

    Parameters
    ----------
    data : np.ndarray
        Calcium imaging data.
    shuffled_data : np.ndarray, optional
        Precomputed shuffled data.
    event_data : np.ndarray, optional
        Event trace to generate shuffle if needed.
    neuron_index : int, optional
        Index of neuron to plot. Random if None.
    data_color : str
        Line color for original trace.
    shuffle_color : str
        Line color for shuffled trace.
    show_plot : bool
        Display the plot.
    save_plot : bool
        Save the plot to file.
    save_path : str
        Destination path if saving.
    save_format : str
        Format of the saved figure.
    dpi : int
        Resolution of saved figure.
    **kwargs : dict
        Additional plot arguments.

    Returns
    -------
    None
    """
    data = _input_validator(data=data)
    if neuron_index is None:
        neuron_index = random.randint(1, np.shape(data)[0] - 1)
    else:
        neuron_index += 1  # Adjusted to accommodate time row
    if shuffled_data is not None:
        shuffled_neuron = shuffled_data[neuron_index, :]
    else:
        if event_data is not None:
            shuffled_data = generate_event_shuffle(data=data, event_data=event_data)
            shuffled_neuron = shuffled_data[neuron_index, :]
        else:
            # Perform OASIS deconvolution
            decon_trace, event_trace = deconvolve(data[neuron_index, :], sn=0.25, penalty=0)[0:2]
            # Binarize event trace
            event_trace[event_trace < 1] = 0
            event_trace[event_trace >= 1] = 1

            shuffled_neuron = generate_event_shuffle(data=np.vstack((data[0, :], decon_trace)),
                                                     event_data=np.vstack((data[0, :], event_trace)))

    plt.figure(figsize=(10, 5))
    plt.subplot(211)
    plt.plot(data[0, :], data[neuron_index, :], c=data_color, label='ground truth', **kwargs)
    plt.ylabel('ΔF/F')
    plt.legend(loc='upper left')
    plt.subplot(212)
    plt.plot(data[0, :], shuffled_neuron[1, :], c=shuffle_color, label='shuffled', **kwargs)
    plt.ylabel('')
    plt.ylabel('ΔF/F')
    plt.xlabel('Time')
    plt.legend(loc='upper left')
    if save_plot:
        if save_path is None:
            save_path = os.getcwd() + f'fig'
        elif not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(fname=save_path, bbox_inches='tight', dpi=dpi, format=save_format)
    if show_plot:
        plt.show()


def plot_correlation_hist(data=None,
                          corr_matrices=None,
                          method="pearson",
                          colors=None,
                          labels=None,
                          title=None,
                          ylabel="Frequency",
                          xlabel="Correlation coefficient",
                          alpha=0.3,
                          bin_width=0.01,
                          show_plot=True,
                          save_plot=False,
                          save_path=None,
                          dpi=300,
                          save_format='png',
                          **kwargs):
    """
    Plot histograms of correlation values for one or more datasets or correlation matrices.

    Parameters
    ----------
    data : list of np.ndarray, optional
        List of calcium datasets to compute correlation matrices.
    corr_matrices : list of np.ndarray, optional
        Precomputed correlation matrices.
    method : str, default='pearson'
        Correlation method to use if data is supplied.
    colors : list of str, optional
        Histogram colors.
    labels : list of str, optional
        Labels for each dataset/matrix.
    title : str
        Plot title.
    ylabel : str
        Y-axis label.
    xlabel : str
        X-axis label.
    alpha : float
        Transparency for histogram bars.
    bin_width : float
        Histogram bin width.
    show_plot : bool
        Whether to display the plot.
    save_plot : bool
        Whether to save the figure.
    save_path : str
        Destination for saved plot.
    dpi : int
        Figure resolution.
    save_format : str
        File format.
    **kwargs : dict
        Extra arguments passed to `plt.hist`.

    Returns
    -------
    None
    """
    if corr_matrices is None and data is None:
        raise ValueError("Either 'data' or 'corr_matrices' must be provided.")

    if corr_matrices is None:
        # Compute correlation matrices from raw data
        corr_matrices = []
        for d in data:
            mat = get_correlation_matrix(data=d, method=method)
            np.fill_diagonal(mat, 0)
            corr_matrices.append(mat)

    num_matrices = len(corr_matrices)

    # Set default colors/labels if needed
    if colors is None:
        colors = [f"C{i}" for i in range(num_matrices)]
    if labels is None:
        labels = [f"Dataset {i + 1}" for i in range(num_matrices)]

    # Plot each matrix histogram
    for mat, color, label in zip(corr_matrices, colors, labels):
        vals = mat[np.triu_indices_from(mat, k=1)]
        bins = int(np.ceil((vals.max() - vals.min()) / bin_width))
        plt.hist(vals, bins=bins, color=color, alpha=alpha, label=label, **kwargs)

    if labels:
        plt.legend(loc='upper left')
    if title:
        plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)

    if show_plot:
        plt.show()

    if save_plot:
        import os
        if save_path is None:
            save_path = os.path.join(os.getcwd(), f"correlation_hist.{save_format}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(fname=save_path, bbox_inches='tight', dpi=dpi, format=save_format)
