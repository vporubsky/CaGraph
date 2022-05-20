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
from setup import FC_DATA_PATH
from dg_network_graph import DGNetworkGraph as nng
import numpy as np
import matplotlib.pyplot as plt
import sklearn

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
    print(time)
    for row in range(np.shape(data)[0]):
        np.random.shuffle(data[row, :])
    data[0, :] = time.copy()
    print(data[0,:])
    print(time)
    return data

def generate_randomized_timeseries_binned(data: list) -> np.ndarray:
    """
    data: list

    Parameter data should contain a list of np.ndarray objects.

    Return a numpy array or NWB file.
    """
    time = data[0, :].copy()
    build_new_array = np.ndarray()
    # build shuffled dist
    for row in range(np.shape(data)[0]):
    for row in range(np.shape(data)[0]):
        np.random.shuffle(data[row, :])
    data[0, :] = time.copy()
    print(data[0,:])
    print(time)
    return data

data = np.genfromtxt(FC_DATA_PATH + '/2-1_D1_smoothed_calcium_traces.csv', delimiter=',')
random_binned_data = generate_randomized_timeseries_binned(data=data.copy())
random_data = generate_randomized_timeseries_matrix(data=data.copy())
# Check if context A only random networks are different than context B only, and how they compare to
np.savetxt('2-1_random_binned_data.csv', random_binned_data, delimiter=",")
np.savetxt('2-1_random_data.csv', random_data, delimiter=",")

#%%
plt.plot(data[0, :], random_data[4, :])
plt.show()

plt.imshow(random_data[:,0:800], cmap='hot', interpolation='nearest')
plt.show()


#%%
random_nng = nng('2-1_random_data.csv')
y=random_nng.pearsons_correlation_matrix
np.fill_diagonal(y, 0)
plt.imshow(y, cmap='hot', interpolation='nearest')
plt.show()

print(f"The max Pearson's r-value is: {np.max(y)}")