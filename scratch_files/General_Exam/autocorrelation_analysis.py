'''Check data for autocorrelations and remove influence of autocorrelations from connectivity analysis.'''
import os

from dg_network_graph import DGNetworkGraph as nng
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm

day1_untreated = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv', '1055-3_D1_smoothed_calcium_traces.csv', '1055-4_D1_smoothed_calcium_traces.csv', '14-0_D1_smoothed_calcium_traces.csv']
day5_untreated = ['1055-1_D5_smoothed_calcium_traces.csv', '1055-2_D5_smoothed_calcium_traces.csv','1055-3_D5_smoothed_calcium_traces.csv', '1055-4_D5_smoothed_calcium_traces.csv', '14-0_D5_smoothed_calcium_traces.csv']
day9_untreated = ['1055-1_D9_smoothed_calcium_traces.csv', '1055-2_D9_smoothed_calcium_traces.csv','1055-3_D9_smoothed_calcium_traces.csv', '1055-4_D9_smoothed_calcium_traces.csv', '14-0_D9_smoothed_calcium_traces.csv']

path_to_data = '/LC-DG-FC-data/'
path_to_export = '/scratch_files/General_Exam/'

file = day1_untreated[0]
nn = nng(path_to_data + day1_untreated[0])
mouse_id = file.replace('_D1_smoothed_calcium_traces.csv', '')
threshold = 0.5

conA = nn.get_context_A_graph(threshold=threshold)
conB = nn.get_context_B_graph(threshold=threshold)

row_idx = 24
sample_idx = 0

data = pd.read_csv(path_to_data + day1_untreated[sample_idx])
row_conA = data.iloc[row_idx,0:1800]
plt.plot(np.linspace(0,180, 1800), row_conA)
plt.ylabel('Context A Ca Fluorescence')
plt.show()

sm.graphics.tsa.plot_acf(row_conA, lags = 1000)
plt.ylabel('Context A Pearsons -- ACF')
plt.show()

sm.graphics.tsa.plot_pacf(row_conA, lags = 1000)
plt.ylabel('Context A -- PACF')
plt.show()

row_conB = data.iloc[row_idx,1800:3600]
plt.plot(np.linspace(180,360, 1800), row_conB)
plt.ylabel('Context B Ca Fluorescence')
plt.show()

sm.graphics.tsa.plot_acf(row_conB, lags = 1000)
plt.ylabel('Context B Pearsons -- ACF')
plt.show()

sm.graphics.tsa.plot_pacf(row_conB, lags = 1000)
plt.ylabel('Context B -- PACF')
plt.show()

#%% compute the average autocorrelation for an array of timeseries data
import statsmodels.api as sm
import os
export_path = '/scratch_files/General_Exam/'
import_data_path = os.getcwd()+'/LC-DG-FC-data/'
dpi = 200
file = day1_untreated[1]
nn = nng(path_to_data + day1_untreated[1])
mouse_id = file.replace('_D1_smoothed_calcium_traces.csv', '')
ns_idx, conA_idx, conB_idx = nn.get_context_active(
        import_data_path + mouse_id + f'_D1_neuron_context_active.csv')
threshold = 0.3

num_lags = 100
dynamics = nn.context_A_dynamics[conB_idx, :]

store_acf = sm.tsa.acf(dynamics[0,:], nlags=500)
for idx in range(np.shape(dynamics)[0]):
    if idx != 0:
        autocorrelations = sm.tsa.acf(dynamics[idx,:], nlags=500)
        if np.isnan(autocorrelations).any():
            continue
        else:
            store_acf = np.vstack((store_acf, autocorrelations))

#%%


avg_acf = np.mean(store_acf, axis=0)
x = np.linspace(0, len(autocorrelations), len(autocorrelations))
plt.plot(x, avg_acf, 'o')
plt.show()