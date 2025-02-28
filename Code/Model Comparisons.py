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

# Removing the last few extra rows
df=df.head(9351)
#%%
print(df['T'].iloc[-20:])


df.replace(-200, pd.NA, inplace=True)

df.fillna(method='ffill', inplace=True)

print(df.head(20))
# df.to_csv('C:/Users/saira/OneDrive/Desktop/Temporai/Deep Sybil/External Data/air quality.csv')

#%%
df.drop(columns=['Unnamed: 15', 'Unnamed: 16'], inplace=True)

df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')

df.drop(columns=['Date', 'Time'], inplace=True)

print(df['T'].iloc[-20:])
#%%
import numpy as np
import matplotlib.pyplot as plt

def average_forecast(data, h):
    """Average model: Forecast is the mean of all observations."""
    mean_value = np.mean(data)
    return [mean_value] * h


def naive_forecast(data, h):
    """Naive model: Forecast is the last observed value."""
    last_value = data[-1]
    return [last_value] * h


def drift_forecast(data, h):
    """Drift model: Linear trend between first and last observation."""
    slope = (data[-1] - data[0]) / (len(data) - 1)
    return [data[-1] + i * slope for i in range(1, h + 1)]


def ses_forecast(data, h, alpha=0.3):
    """Simple Exponential Smoothing: Weighted average of past observations."""
    forecast = [data[0]]  # Initial forecast
    for t in range(1, len(data)):
        forecast.append(alpha * data[t] + (1 - alpha) * forecast[-1])
    return [forecast[-1]] * h


def seasonal_naive_forecast(data, h, seasonal_period=24):
    """
    Seasonal Naive model: Forecast is the value from the same point in the previous seasonal cycle.

    Args:
        data: Array of historical data.
        h: Number of steps ahead to forecast.
        seasonal_period: The length of the seasonal cycle (e.g., 24 for hourly data with daily seasonality).

    Returns:
        List of forecasted values.
    """
    forecasts = []
    for i in range(h):
        forecasts.append(data[-seasonal_period + (i % seasonal_period)])
    return forecasts

def holt_winters_forecast(data, h, alpha=0.005, beta=0.5, gamma=0.2, season_period=24):
    """
    Holt-Winters: Handles level, trend, and seasonality using triple exponential smoothing.
    Args:
        data: Array of time series data.
        h: Number of steps ahead to forecast.
        alpha: Smoothing parameter for level.
        beta: Smoothing parameter for trend.
        gamma: Smoothing parameter for seasonality.
        season_period: The seasonal period (e.g., 24 for hourly data with daily seasonality).
    Returns:
        List of forecasts for h steps ahead.
    """
    n = len(data)
    if n < season_period:
        raise ValueError("Length of data must be greater than or equal to the seasonal period.")

    # Initialize components
    level = data[0]
    trend = data[1] - data[0]
    seasonal = [data[i] - level for i in range(season_period)]  # Initial seasonal indices

    forecasts = []

    # Update components iteratively
    for t in range(n):
        if t >= season_period:
            seasonality = seasonal[t % season_period]
        else:
            seasonality = seasonal[t]  # Use the initialized values for the first season_period steps

        prev_level = level
        level = alpha * (data[t] - seasonality) + (1 - alpha) * (prev_level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        seasonal[t % season_period] = gamma * (data[t] - level) + (1 - gamma) * seasonal[t % season_period]

    # Forecast h-steps ahead
    for i in range(1, h + 1):
        forecast = level + i * trend + seasonal[(n + i - 1) % season_period]
        forecasts.append(forecast)

    return forecasts

#%%
def box_one_step_forecast(b,f,c,d,u,y,steps):

    y1=[]
    y2=[]
    preds = []
    residuals = []
    b_order=len(b)
    f_order=len(f)
    c_order=len(c)
    d_order=len(d)

    d = np.concatenate(([1], d))
    c = np.concatenate(([1], c))
    f = np.concatenate(([1], f))

    bd = np.convolve(b, d, "full")
    cf = np.convolve(c, f, "full")

    df = np.convolve(d, f, "full")
    cf = cf[1:]
    df = df[1:]
    df = [-1 * i for i in df]
    # print(bd)
    # print(cf)
    # print(df)



    bd=np.convolve(b,d,"full")

    # Iterate over the data for the given steps
    for i in range(steps):
        # Compute AR and MA terms

        u_term = sum(bd[j] * (u[i - j] if i - j >= 0 else 0) for j in range(len(bd)))
        e_term = sum(cf[j] * (residuals[i - j-1] if i - j-1 >= 0 else 0) for j in range(len(cf)))
        y_term = sum(df[j] * (y[i - j-1] if i - j-1 >= 0 else 0) for j in range(len(df)))

        pred = u_term+e_term+y_term

        preds.append(pred)
        residual = y[i] - pred if i < len(y) else 0
        residuals.append(residual)



    return preds,residuals


def hstep_forecast(data, ar_params, ma_params, steps):
    """
    Generate h-step forecasts dynamically using AR and MA parameters.

    Parameters:
    - data: array-like, time series data
    - ar_params: array-like, AR parameters
    - ma_params: array-like, MA parameters
    - steps: int, number of forecast steps
    - start_step: int, starting point in the dataset for forecasting

    Returns:
    - preds: list of forecasts
    - forecast_errors: list of forecast errors
    """
    preds = []
    forecast_errors = []
    ar_order = len(ar_params)
    ma_order = len(ma_params)
    one_step_preds=[]
    one_step_residuals=[]

    data_size=len(data)


    for i in range(data_size):

        # Compute AR and MA terms
        ar_term = sum(ar_params[j] * (data[i - j - 1] if i - j - 1 >= 0 else 0) for j in range(ar_order))
        ma_term = sum(ma_params[j] * (one_step_residuals[i - j - 1] if i - j - 1 >= 0 else 0) for j in range(ma_order))

        # One-step prediction
        pred = ar_term + ma_term
        one_step_preds.append(pred)

        # Compute residual
        residual = data[i] - pred if i < len(data) else 0
        one_step_residuals.append(residual)

    # Iterate for the number of steps
    for i in range(steps):
        # AR term calculation
        ar_term = sum(
            ar_params[j] * (preds[i - j - 1] if i - j - 1 >= 0 else data[data_size + i - j - 1])
            for j in range(ar_order)
        )

        # MA term calculation
        ma_term = sum(
            ma_params[j] * (one_step_residuals[data_size + i - j - 1] if i - j - 1 < 0 else 0)
            for j in range(ma_order)
        )

        # Forecast for the current step
        pred = ar_term + ma_term
        preds.append(pred)

    return preds


#%%
def box_h_step_forecast(b, f, c, d, u, y, steps):
    """
    Perform h-step ahead forecasts given model parameters and input data, following the rules from the image.

    Parameters:
    - b: Coefficients for input series u (AR part).
    - f: Coefficients for residuals (MA part).
    - c: Coefficients for past residuals (cross-term part).
    - d: Coefficients for past values of y (AR part).
    - u: Input series (numpy array).
    - y: Observed series (numpy array).
    - steps: Number of steps to forecast ahead.
    - start_step: Index indicating when to begin h-step forecasting.

    Returns:
    - preds: Forecasted values (list).
    - residuals: Residuals of the forecasts (list).
    """
    preds = []  # Store predicted values
    residuals = []  # Store residuals of predictions
    start_step=len(y)
    one_step_preds, one_step_residuals = box_one_step_forecast(b, f, c, d, u, y_train, len(y_train))
    # Extend coefficients to include initial terms
    d = np.concatenate(([1], d))
    c = np.concatenate(([1], c))
    f = np.concatenate(([1], f))

    # Compute convolutions
    bd = np.convolve(b, d, "full")
    cf = np.convolve(c, f, "full")
    df = np.convolve(d, f, "full")

    cf = cf[1:]
    df = df[1:]
    df = [-1 * i for i in df]

    # Print convolutions for debugging purposes
    # print("bd (input * AR part):", bd)
    # print("cf (residual * MA part):", cf)
    # print("df (past y * AR * MA):", df)
    print(one_step_residuals[-1])
    # Iterate for the specified number of steps
    for i in range(steps):
        # print(u)
        u_term = sum(bd[j] * (u[start_step+i - j]) for j in range(len(bd)))
        # print(u_term)
        # print("u done")
        e_term = sum(cf[j] * (one_step_residuals[start_step+i - j - 1] if i - j - 1 < 0 else 0) for j in range(len(cf)))
        # print(e_term)
        # print("e done")
        y_term = sum(df[j] * (y[start_step+i - j - 1] if i - j - 1 < 0 else preds[i - j - 1]) for j in range(len(df)))
        # print(y_term)
        # print("y done")
        pred = u_term + e_term + y_term
        preds.append(pred)
        # print("Pred is",pred)

    return preds

#%%
b = np.load('Best_Model_final3_b_original.npy')
f = np.load('Best_Model_final3_f_original.npy')
c = np.load('Best_Model_final3_c_original.npy')
d = np.load('Best_Model_final3_d_original.npy')

ar_loaded = np.load('Best_Model_ar_original.npy')
ma_loaded = np.load('Best_Model_ma_original.npy')

print(b,f,c,d,ar_loaded,ma_loaded)

#%%
y=df['T'].values
train_size = int(0.8 * len(y))
y_train = y[:train_size]
y_test = y[train_size:]


u=df['AH']
train_size = int(0.8 * len(u))
u_train = u[:train_size]
u_test = u[train_size:]
#%%
target_column = 'T'

# Features (X) and target (y)
X = df.drop(columns=[target_column])
y = df[target_column]
from sklearn.model_selection import train_test_split
# Perform the split: 80% training and 20% testing
X_train, X_test, _,_ = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)


#%%

y=df['T'].values
train_size = int(0.8 * len(y))
y_train = y[:train_size]
y_test = y[train_size:]


u=df['AH']
train_size = int(0.8 * len(u))
u_train = u[:train_size]
u_test = u[train_size:]


h=240



import statsmodels.api as sm
X_train_selected = X_train[['RH', 'AH']]
X_train_selected_const = sm.add_constant(X_train_selected)

X_test_selected = X_test[['RH', 'AH']]
X_test_selected_const = sm.add_constant(X_test_selected)

# Fit the OLS model
model = sm.OLS(y_train, X_train_selected_const).fit()

# Generate predictions for the test set
mlr_pred = model.predict(X_test_selected_const[:h]).values
average_pred = average_forecast(y_train, h)
naive_pred = naive_forecast(y_train, h)
drift_pred = drift_forecast(y_train, h)
ses_pred = ses_forecast(y_train, h)
hw_pred = holt_winters_forecast(y_train, h)
seasonal_naive_pred = seasonal_naive_forecast(y_train, h, seasonal_period=24)

sarima_pred=hstep_forecast(y_train,ar_loaded,ma_loaded,h)
box_pred=box_h_step_forecast(b,f,c,d,u,y_train,h)

actuals=y_test[:h]

plt.figure(figsize=(12, 8))

# Plot each prediction against actuals
plt.plot(actuals, label='Actuals', color='black', linewidth=2)
plt.plot(average_pred, label='Average Forecast', linestyle='--')
plt.plot(naive_pred, label='Naive Forecast', linestyle='--')
plt.plot(drift_pred, label='Drift Forecast', linestyle='--')
plt.plot(ses_pred, label='SES Forecast', linestyle='--')
plt.plot(hw_pred, label='Holt-Winters Forecast', linestyle='--')
plt.plot(mlr_pred, label='Multiple Linear Regression Forecast', linestyle='--')

# plt.plot(seasonal_naive_pred, label='Seasonal Naive Forecast', linestyle='--')
plt.plot(sarima_pred, label='SARIMA Forecast', linestyle='--')
plt.plot(box_pred, label='Box Forecast', linestyle='--')

# Adding titles and labels
plt.title('Comparison of Forecasting Models', fontsize=16)
plt.xlabel('Time Steps', fontsize=14)
plt.ylabel('Values', fontsize=14)

# Add legend
plt.legend(loc='upper left', fontsize=10)

# Show grid for better readability
plt.grid(alpha=0.4)

# Display the plot
plt.show()

from sklearn.metrics import mean_squared_error

# Calculate MSE and SSE for each prediction
def calculate_metrics(actuals, predictions):
    mse = mean_squared_error(actuals, predictions)
    sse = mse * len(actuals)  # SSE = MSE * number of samples
    return mse, sse

# Compute metrics for each model
metrics = {
    "Average Forecast": calculate_metrics(actuals, average_pred),
    "Naive Forecast": calculate_metrics(actuals, naive_pred),
    "Drift Forecast": calculate_metrics(actuals, drift_pred),
    "SES Forecast": calculate_metrics(actuals, ses_pred),
    "Holt-Winters Forecast": calculate_metrics(actuals, hw_pred),
    "Linear Regression Forecast": calculate_metrics(actuals,mlr_pred),
    "Seasonal Naive Forecast": calculate_metrics(actuals, seasonal_naive_pred),
    "SARIMA Forecast": calculate_metrics(actuals, sarima_pred),
    "Box Forecast": calculate_metrics(actuals, box_pred),
}

# Print the results
print(f"{'Model':<30} {'MSE':<15} {'SSE':<15}")
print("-" * 60)
for model_name, (mse, sse) in metrics.items():
    print(f"{model_name:<30} {mse:<15.5f} {sse:<15.5f}")

#%%
# Just showing base models

plt.figure(figsize=(12, 8))

# Plot each prediction against actuals
plt.plot(actuals, label='Actuals', color='black', linewidth=2)
plt.plot(average_pred, label='Average Forecast', linestyle='--')
plt.plot(naive_pred, label='Naive Forecast', linestyle='--')
plt.plot(drift_pred, label='Drift Forecast', linestyle='--')
plt.plot(ses_pred, label='SES Forecast', linestyle='--')
# plt.plot(hw_pred, label='Holt-Winters Forecast', linestyle='--')
# plt.plot(mlr_pred, label='Multiple Linear Regression Forecast', linestyle='--')

plt.plot(seasonal_naive_pred, label='Seasonal Naive Forecast', linestyle='--')
# plt.plot(sarima_pred, label='SARIMA Forecast', linestyle='--')
# plt.plot(box_pred, label='Box Forecast', linestyle='--')

plt.title('Comparison of Base Models', fontsize=16)
plt.xlabel('Time Steps', fontsize=14)
plt.ylabel('Values', fontsize=14)
plt.legend(loc='upper left', fontsize=10)
plt.grid(alpha=0.4)
plt.show()


#%%
# Just showing base models

plt.figure(figsize=(12, 8))

# Plot each prediction against actuals
plt.plot(actuals, label='Actuals', color='black', linewidth=2)
# plt.plot(average_pred, label='Average Forecast', linestyle='--')
# plt.plot(naive_pred, label='Naive Forecast', linestyle='--')
# plt.plot(drift_pred, label='Drift Forecast', linestyle='--')
# plt.plot(ses_pred, label='SES Forecast', linestyle='--')
plt.plot(hw_pred, label='Holt-Winters Forecast', linestyle='--')
plt.plot(mlr_pred, label='Multiple Linear Regression Forecast', linestyle='--')

# plt.plot(seasonal_naive_pred, label='Seasonal Naive Forecast', linestyle='--')
plt.plot(sarima_pred, label='SARIMA Forecast', linestyle='--')
plt.plot(box_pred, label='Box Forecast', linestyle='--')

plt.title('Comparison of Advanced Models', fontsize=16)
plt.xlabel('Time Steps', fontsize=14)
plt.ylabel('Values', fontsize=14)
plt.legend(loc='upper left', fontsize=10)
plt.grid(alpha=0.4)
plt.show()

#%%
# Just Showing Holt Winters
plt.figure(figsize=(12, 8))

# Plot each prediction against actuals
plt.plot(actuals, label='Actuals', color='black', linewidth=2)
# plt.plot(average_pred, label='Average Forecast', linestyle='--')
# plt.plot(naive_pred, label='Naive Forecast', linestyle='--')
# plt.plot(drift_pred, label='Drift Forecast', linestyle='--')
# plt.plot(ses_pred, label='SES Forecast', linestyle='--')
plt.plot(hw_pred, label='Holt-Winters Forecast', linestyle='--')
# plt.plot(mlr_pred, label='Multiple Linear Regression Forecast', linestyle='--')

# plt.plot(seasonal_naive_pred, label='Seasonal Naive Forecast', linestyle='--')
# plt.plot(sarima_pred, label='SARIMA Forecast', linestyle='--')
# plt.plot(box_pred, label='Box Forecast', linestyle='--')
#
plt.title('Holt Winters Model', fontsize=16)
plt.xlabel('Time Steps', fontsize=14)
plt.ylabel('Values', fontsize=14)
plt.legend(loc='upper left', fontsize=10)
plt.grid(alpha=0.4)
plt.show()


#%%
# Top 4 best models
# Just Showing Holt Winters
plt.figure(figsize=(12, 8))

# Plot each prediction against actuals
plt.plot(actuals, label='Actuals', color='black', linewidth=2)
# plt.plot(average_pred, label='Average Forecast', linestyle='--')
# plt.plot(naive_pred, label='Naive Forecast', linestyle='--')
# plt.plot(drift_pred, label='Drift Forecast', linestyle='--')
# plt.plot(ses_pred, label='SES Forecast', linestyle='--')
plt.plot(hw_pred, label='Holt-Winters Forecast', linestyle='--')
# plt.plot(mlr_pred, label='Multiple Linear Regression Forecast', linestyle='--')

plt.plot(seasonal_naive_pred, label='Seasonal Naive Forecast', linestyle='--')
plt.plot(sarima_pred, label='SARIMA Forecast', linestyle='--')
plt.plot(box_pred, label='Box Forecast', linestyle='--')
#
plt.title('Top 4 best models', fontsize=16)
plt.xlabel('Time Steps', fontsize=14)
plt.ylabel('Values', fontsize=14)
plt.legend(loc='upper left', fontsize=10)
plt.grid(alpha=0.4)
plt.show()
