'''This Python script is used to practice implementing ARIMA models for possible
application to pre-whitening timeseries data.'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

path_to_data ='/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/LC-DG-FC-data/'
path_to_export = '/Users/veronica_porubsky/GitHub/DG_fear_conditioning_graph_theory/scratch_files/General_Exam/'

df = pd.read_csv(path_to_data + '14-0_D1_smoothed_calcium_traces.csv')
plt.xlabel('Time')
plt.ylabel('Calcium fluorescence')
plt.plot(np.linspace(0,360,3600), df.iloc[1])
plt.show()

rolling_mean = df.iloc[1].rolling(window=500).mean().dropna()
rolling_std = df.iloc[1].rolling(window=500).std().dropna()
plt.plot(np.linspace(0, 360, 3600), df.iloc[1], color = 'blue', label = 'Original')
plt.plot(np.linspace(50, 360, 3101), rolling_mean, color = 'red', label = 'Rolling Mean')
plt.plot(np.linspace(50, 360, 3101), rolling_std, color = 'black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean and Standard Deviation')
plt.show()


result = adfuller(df.iloc[1])

print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
print('Critical values:')
for key, value in result[4].items():
    print(f'\t{key}: {value}')


def get_stationarity(timeseries, window_size):
    # rolling statistics
    rolling_mean = timeseries.rolling(window=window_size).mean().dropna()
    rolling_std = timeseries.rolling(window=window_size).std().dropna()

    # rolling statistics plotting
    original = plt.plot(np.linspace(0, len(timeseries)/10, len(timeseries)), timeseries, color='blue', label='Original')
    mean = plt.plot(np.linspace(window_size/10, len(timeseries)/10, len(rolling_mean)), rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(np.linspace(window_size/10, len(timeseries)/10, len(rolling_std)), rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show(block=False)

    # Dickey-Fuller test:
    result = adfuller(timeseries)
    print(f'ADF statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

original = get_stationarity(df.iloc[1], window_size=500)

# Effect of subtracting the rolling mean
rolling_mean = df.iloc[1].rolling(window=500).mean()
df_minus_mean = df.iloc[1] - rolling_mean
df_minus_mean.dropna(inplace=True)
get_stationarity(df_minus_mean, window_size=500)

# Effect of time shifting
df_shift = df.iloc[1] - df.iloc[1].shift()
df_shift.dropna(inplace=True)
get_stationarity(df_shift, window_size=500)


# Building an ARIMA model
df = pd.read_csv('14-0_D1_all_calcium_traces.csv').T
# use Partial Auto-correlation Function to determine the best order of the AR (auto-regression) model
sm.graphics.tsa.plot_pacf(df_shift, lags=100)
plt.ylabel('PACF')
plt.show()

# use Auto-correlation Function to determine order of the MA (moving average) model
sm.graphics.tsa.plot_acf(df_shift, lags=100)
plt.ylabel('ACF')
plt.show()

# build the model
original_timeseries = pd.DataFrame(index=pd.date_range(start='1/1/2020', periods=len(df[1]), freq='D'))
original_timeseries[''] = list(df[1])
timeseries = pd.DataFrame(index=pd.date_range(start='1/1/2020', periods=len(df_shift), freq='D')) # necessary for ARIMA
timeseries[''] = list(df_shift)
decomposition = seasonal_decompose(timeseries)
model = ARIMA(timeseries, order=(10,1,20)) #Todo: determine how to set these
results = model.fit()
plt.plot(original_timeseries, color='blue')
plt.plot(results.fittedvalues, color='red')
plt.show()

# See how model compares to orginal time series
forecasted_results=  results.forecast(steps=1000)
plt.plot(forecasted_results)