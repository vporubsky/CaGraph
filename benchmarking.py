"""
Developer Name: Veronica Porubsky
Developer ORCID: 0000-0001-7216-3368
Developer GitHub Username: vporubsky
Developer Email: verosky@uw.edu

File Creation Date: 
File Final Edit Date:

Description: A utilities package to assist with analyses

Todo: eventually this will need to be incorporated into a formal submodule
"""
# Import packages
import numpy as np

# Todo: rename submodules
#%% Preprocess data if not completed
# Todo: look up smoothing algorithm for calcium data
def smooth(data):
    """

    Smooth unprocessed data to remove noise.
    """
    smoothed_data = data
    return smoothed_data

# Todo: make auto-clean option for those that don't have experience
def auto_preprocess(data):

    preprocessed_data = smooth(data)
    return preprocessed_data
# %% Clean up data
def remove_quiescent(data, event_data, event_num_threshold=5):
    """
    data: numpy.ndarray
    event_bins: numpy.ndarray

    Removes inactive neurons from the dataset
    """
    binarized_event_data = np.where(event_data > 0.0005, 1, 0)
    new_event_data = np.zeros((1, np.shape(event_data)[1]))
    new_data = np.zeros((1, np.shape(data)[1]))
    for row in range(np.shape(binarized_event_data)[0]):
        if np.sum(binarized_event_data[row, :]) <= event_num_threshold:
            continue
        else:
            new_event_data = np.vstack((new_event_data, event_data[row,:]))
            new_data = np.vstack((new_data, data[row,:]))
    return new_data[1:, :], new_event_data[1:,:]


def count_sign_switch(row_data):
    subtract = row_data[0:len(row_data)-1] - row_data[1:]
    a = subtract
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    return np.sum(signchange)


def remove_low_activity(data, event_data, event_num_threshold=5):

    #apply activity treshold
    new_event_data = np.zeros((1, np.shape(event_data)[1]))
    new_data = np.zeros((1, np.shape(data)[1]))
    for row in range(np.shape(data)[0]):
        if count_sign_switch(row_data = data[row,:]) <= 5 and not row == 0:
            continue
        else:
            new_event_data = np.vstack((new_event_data, event_data[row,:]))
            new_data = np.vstack((new_data, data[row,:]))
    return new_data[1:, :], new_event_data[1:,:]

# %% Suitability for graph theory analysis
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
    threshold = 0.01
    flat_random_binned_list = [0 if value < threshold else value for value in flat_random_binned_list]
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
        binned_row = event_bins(data=data[row + 1, :], events=event_data[row + 1, :])
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


#%% Randomize across all

def generate_randomized_across_population(data: np.ndarray, event_data: np.ndarray) -> np.ndarray:

    # First split all the data and make a long list of
    return