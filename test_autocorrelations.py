'''Check data for autocorrelations and remove influence of autocorrelations from connectivity analysis.'''
from neuronal_network_graph import neuronal_network_graph as nng
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm

day1_untreated = ['1055-1_D1_smoothed_calcium_traces.csv', '1055-2_D1_smoothed_calcium_traces.csv', '1055-3_D1_smoothed_calcium_traces.csv', '1055-4_D1_smoothed_calcium_traces.csv', '14-0_D1_smoothed_calcium_traces.csv']
day5_untreated = ['1055-1_D5_smoothed_calcium_traces.csv', '1055-2_D5_smoothed_calcium_traces.csv','1055-3_D5_smoothed_calcium_traces.csv', '1055-4_D5_smoothed_calcium_traces.csv', '14-0_D5_smoothed_calcium_traces.csv']
day9_untreated = ['1055-1_D9_smoothed_calcium_traces.csv', '1055-2_D9_smoothed_calcium_traces.csv','1055-3_D9_smoothed_calcium_traces.csv', '1055-4_D9_smoothed_calcium_traces.csv', '14-0_D9_smoothed_calcium_traces.csv']

nn = nng(day1_untreated[1])
threshold = 0.5

conA = nn.get_context_A_graph(threshold=threshold)
conB = nn.get_context_B_graph(threshold=threshold)

row_idx = 6
sample_idx = 0

data = pd.read_csv(day1_untreated[sample_idx])
row_conA = data.iloc[row_idx,0:1800]
plt.plot(np.linspace(0,180, 1800), row_conA)
plt.ylabel('Context A Ca Fluorescence')
plt.show()

sm.graphics.tsa.plot_acf(row_conA, lags = 100)
plt.ylabel('Context A Pearsons -- ACF')
plt.show()

sm.graphics.tsa.plot_pacf(row_conA, lags = 100)
plt.ylabel('Context A -- PACF')
plt.show()

row_conB = data.iloc[row_idx,1800:3600]
plt.plot(np.linspace(180,360, 1800), row_conB)
plt.ylabel('Context B Ca Fluorescence')
plt.show()

sm.graphics.tsa.plot_acf(row_conB, lags = 100)
plt.ylabel('Context B Pearsons -- ACF')
plt.show()

sm.graphics.tsa.plot_pacf(row_conB, lags = 100)
plt.ylabel('Context B -- PACF')
plt.show()