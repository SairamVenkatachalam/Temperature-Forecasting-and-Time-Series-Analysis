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


def compare_models(data, train_size, h):
    """
    Compare forecasting models and plot predictions only for the test data period.
    Args:
        data: Array of time series data.
        train_size: Proportion of data to use for training (0 < train_size < 1).
        h: Number of steps ahead to forecast.
    Returns:
        Dictionary of forecasts for each model, along with SSE for each model.
    """
    # Train-test split
    split_index = int(len(data) * train_size)
    train = data[:split_index]
    test = data[split_index:split_index + h]

    # Generate forecasts
    average_pred = average_forecast(train, h)
    naive_pred = naive_forecast(train, h)
    drift_pred = drift_forecast(train, h)
    ses_pred = ses_forecast(train, h)
    hw_pred = holt_winters_forecast(train, h)
    seasonal_naive_pred = seasonal_naive_forecast(train, h, seasonal_period=24)

    models = {
        "Average": average_pred,
        "Naive": naive_pred,
        "Drift": drift_pred,
        "SES": ses_pred,
        "Seasonal Naive": seasonal_naive_pred,
        "Holt-Winters": hw_pred
    }
    sse = {model: np.sum((test - np.array(pred))**2) for model, pred in models.items()}

    # Plot the forecasts against the first 100 test points
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(train), len(train) + len(test)), test, label='Test Data', color='black', linewidth=2)

    # Limit plot to first 100 test points
    future_index = range(len(train), len(train) + min(100, h))
    for model, pred in models.items():
        plt.plot(future_index, pred[:100], label=f'{model} Forecast', linestyle='--')

    plt.title(f"Forecasting Base Models Comparison ")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()

    return {
        "SSE": sse,
        "Forecasts": models
    }

train_size = 0.8
h = 100  # Number of steps ahead to display n the chart
results = compare_models(df['T'].values, train_size, h)

# Display SSE for each model
print("Sum of Squared Errors (SSE) on Test Data:")
for model, error in results["SSE"].items():
    print(f"{model}: {error:.2f}")
