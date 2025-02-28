import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("../toolbox")
import toolbox as ts
from importlib import reload
reload(ts)
reload(ts)

file_path = 'C:/Users/saira/OneDrive/Desktop/GWU Courses/Semester 3/Time Series/Assignments/Time Series Forecasting class/Project/Datasets/Air Quality.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)
print(df.head(20))
#%%


df.replace(-200, pd.NA, inplace=True)

df.fillna(method='ffill', inplace=True)

print(df.head(20))
# df.to_csv('C:/Users/saira/OneDrive/Desktop/Temporai/Deep Sybil/External Data/air quality.csv')

#%%
df.drop(columns=['Unnamed: 15', 'Unnamed: 16'], inplace=True)

df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')

df.drop(columns=['Date', 'Time'], inplace=True)

#%%
import statsmodels.api as sm
import numpy as np
def ACF_PACF_Plot(y, lags, equation=0):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)

    fig = plt.figure()
    lags_range = range(len(acf))

    plt.subplot(211)
    plt.title(f'ACF of Data', fontsize=8)
    markerline, stemlines, baseline = plt.stem(lags_range, acf, linefmt='b-', basefmt=' ', markerfmt='ro')
    markerline.set_markerfacecolor('red')
    n = len(y)
    significance_level = 1.96 / np.sqrt(n)
    plt.fill_between(lags_range, -significance_level, significance_level, color='purple', alpha=0.2,
                     label='Insignificance Region')
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    plt.legend()
    # plt.show()


    # PACF plot
    plt.subplot(212)
    plt.title(f'PACF of Data', fontsize=8)
    markerline, stemlines, baseline = plt.stem(lags_range, pacf, linefmt='b-', basefmt=' ', markerfmt='ro')
    markerline.set_markerfacecolor('red')
    n = len(y)
    significance_level = 1.96 / np.sqrt(n)
    plt.fill_between(lags_range, -significance_level, significance_level, color='purple', alpha=0.2,
                     label='Insignificance Region')
    plt.xlabel('Lags')
    plt.ylabel('Partial Autocorrelation')
    plt.grid(True)
    plt.legend()
    fig.tight_layout(pad=3)
    plt.show()



#%%

def check_stationarity(data):
    print(f"ADF Test for the data")
    ts.ADF_Cal(data)
    ts.kpss_test(data)
    plot_rolling_meanvar(data)

def plot_rolling_meanvar(data):

    rolling_mean = ts.rolling_mean(data)
    rolling_var = ts.rolling_var(data)
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    # Plot rolling mean
    axes[0].plot(rolling_mean.index, rolling_mean, label='Rolling Mean', color='blue')
    axes[0].set_title(f'Rolling Mean of Data')
    axes[0].set_xlabel('Datetime')
    axes[0].set_ylabel('Rolling Mean')
    axes[0].grid()
    axes[0].legend()

    # Plot rolling variance
    axes[1].plot(rolling_var.index, rolling_var, label='Rolling Variance', color='orange')
    axes[1].set_title(f'Rolling Variance of Data')
    axes[1].set_xlabel('Datetime')
    axes[1].set_ylabel('Rolling Variance')
    axes[1].grid()
    axes[1].legend()

    plt.tight_layout()
    plt.show()

#%%
def seasonal_difference(data,season):

    differenced_values = []

    for i in range(season, len(data)):
        differenced_value = data[i] - data[i - season]
        differenced_values.append(differenced_value)

    differenced_series = np.array(differenced_values)

    return differenced_series

#%%
plt.figure()
data=df['T']
plt.plot(df['datetime'],data,label='Temperature',color='#A5D8DD')
plt.title("Temperature Over Time")
plt.ylabel("Temperature")
plt.xlabel("Date")
plt.legend()
plt.show()
#%%
# Checking Stationarity of original data

check_stationarity(df['T'])
ACF_PACF_Plot(df['T'],50)





#%%
from statsmodels.tsa.seasonal import STL
data=df['T']
season=24
differenced_series =seasonal_difference(data,season)

# second time non-seasonal differenign
season=1
differenced_series =seasonal_difference(differenced_series,season)


check_stationarity(differenced_series)
ACF_PACF_Plot(differenced_series,200)

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from sklearn.model_selection import train_test_split

# Doing decomposition on just the train data
train_size = 0.8

train_data, test_data = train_test_split(df, train_size=train_size, shuffle=False)

train_data['timestamp'] = train_data['datetime']
train_data.set_index('timestamp', inplace=True)

epsilon = 1e-6
train_data['T'] = train_data['T'].clip(lower=epsilon)
seasonal_length = 24

# Additive Decomposition
stl_additive = STL(train_data['T'], period=seasonal_length, robust=True).fit()
additive_residual = stl_additive.resid

# Multiplicative Decomposition: Log-transform the data
train_data['T_log'] = np.log(train_data['T'])
stl_multiplicative = STL(train_data['T_log'], period=seasonal_length, robust=True).fit()

multiplicative_residual = train_data['T'] / (np.exp(stl_multiplicative.trend) * np.exp(stl_multiplicative.seasonal) + epsilon)

additive_residual_variance = np.var(additive_residual)
multiplicative_residual_variance = np.var(multiplicative_residual)

fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)

# Additive Decomposition
axes[0, 0].plot(stl_additive.observed, color='blue', label='Observed')
axes[0, 0].set_title('Additive: Observed')
axes[1, 0].plot(stl_additive.seasonal, color='orange', label='Seasonal')
axes[1, 0].set_title('Additive: Seasonal')
axes[2, 0].plot(stl_additive.trend, color='green', label='Trend')
axes[2, 0].set_title('Additive: Trend')
axes[3, 0].plot(additive_residual, color='red', label='Residual')
axes[3, 0].set_title(f'Additive: Residual (Variance = {additive_residual_variance:.4f})')

# Multiplicative Decomposition
axes[0, 1].plot(train_data['T'], color='blue', label='Observed')
axes[0, 1].set_title('Multiplicative: Observed')
axes[1, 1].plot(np.exp(stl_multiplicative.seasonal), color='orange', label='Seasonal')
axes[1, 1].set_title('Multiplicative: Seasonal')
axes[2, 1].plot(np.exp(stl_multiplicative.trend), color='green', label='Trend')
axes[2, 1].set_title('Multiplicative: Trend')
axes[3, 1].plot(multiplicative_residual, color='red', label='Residual')
axes[3, 1].set_title(f'Multiplicative: Residual (Variance = {multiplicative_residual_variance:.4f})')

for ax in axes[:, 0]:  # Additive decomposition
    ax.tick_params(axis='x', rotation=45)
for ax in axes[:, 1]:  # Multiplicative decomposition
    ax.tick_params(axis='x', rotation=45)


plt.tight_layout()
plt.show()

print(f"Additive Decomposition Residual is : {additive_residual_variance:.2f}")
print(f"Multiplicative Decomposition Residual is : {multiplicative_residual_variance:.2f}")

if additive_residual_variance < multiplicative_residual_variance:
    print("Additive STL decomposition is better (lower residual variance).")
else:
    print("Multiplicative STL decomposition is better (lower residual variance).")

#%%
