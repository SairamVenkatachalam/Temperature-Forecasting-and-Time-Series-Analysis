import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("../toolbox")
import toolbox as ts
from importlib import reload
reload(ts)
reload(ts)
import numpy as np
import statsmodels as sm
file_path = 'C:/Users/saira/OneDrive/Desktop/GWU Courses/Semester 3/Time Series/Assignments/Time Series Forecasting class/Project/Datasets/Air Quality.csv'  # Replace with your actual file path
df = pd.read_csv(file_path)

# Removing the last few extra rows
df=df.head(9351)
#%%
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
y=df['T']
train_size = int(0.8 * len(y))
y_train = y[:train_size]
y_test = y[train_size:]

#%%
def seasonal_difference(data,season):

    differenced_values = []

    for i in range(season, len(data)):
        differenced_value = data[i] - data[i - season]
        differenced_values.append(differenced_value)

    differenced_series = np.array(differenced_values)

    return differenced_series
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


def check_stationarity(data):
    print(f"ADF Test for the data")
    ts.ADF_Cal(data)
    ts.kpss_test(data)
    plot_rolling_meanvar(data)


#%%
season=24
differenced_series =seasonal_difference(y_train,season)

# Further differencing non seasonal
season=1
differenced_series =seasonal_difference(differenced_series,season)


check_stationarity(differenced_series)
ACF_PACF_Plot(differenced_series,75)


#%%

def gpac_table(acf_values, j_max=7, k_max=7):
    """
    Code to calculate the GPAC table, given the formulae in the assignment
    I've taken the entire correlation array, and used absolute values to account for negatives
    The first column ie: k=0 has been excluded from the final GPAC table
    """
    gpac_table = np.zeros((j_max+1, k_max+1))

    for k in range(1, k_max + 1):
        for j in range(0, j_max + 1):
            numerator_matrix = np.zeros((k, k))
            denominator_matrix = np.zeros((k, k))

            for row in range(k):
                for col in range(k):
                    numerator_matrix[row, col] = acf_values[abs(j + row - col)]
                    denominator_matrix[row, col] = acf_values[abs(j + row -col)]

            col=k-1
            for row in range(k):
                numerator_matrix[row, col] = acf_values[abs(j + row + 1)]

            numerator_det = np.linalg.det(numerator_matrix)
            denominator_det = np.linalg.det(denominator_matrix)

            tolerance = 10 ** -20

            if abs(numerator_det) < tolerance:
                numerator_det = 0

            if abs(denominator_det) < tolerance:
                denominator_det = 0


            # print(numerator_det,denominator_det)
            # Avoid division by zero
            if denominator_det != 0:
                gpac_table[j,k] = numerator_det / denominator_det
            else:
                gpac_table[j,k] = np.nan
            # print(f" GPAC value at {j},{k} is {gpac_table[j,k]}")
    # print(gpac_table)
    gpac_df = pd.DataFrame(gpac_table, index=[f"j={i}" for i in range(j_max + 1)],
                           columns=[f"k={i}" for i in range(k_max + 1)]).drop(columns="k=0")
    return gpac_df



#%%
import statsmodels.api as sm
import seaborn as sns
autocorr_values = sm.tsa.acf(differenced_series,nlags=200)
gpac_df = gpac_table(autocorr_values, 60,60)



plt.figure(figsize=(10, 6))
sns.heatmap(gpac_df.iloc[20:30, 0:10], annot=True, fmt=".2f", linewidths=0.5,
            center=0, cbar_kws={'label': 'GPAC Value'})

plt.title(f"GPAC Heatmap for Data")
plt.show()

#%%
def one_step_pred(data, ar_params, ma_params, steps,constant=0):
    """
    Generate one-step predictions for the first `steps` observations.

    Parameters:
    - data: array-like, time series data
    - ar_params: array-like, AR parameters
    - ma_params: array-like, MA parameters
    - steps: int, number of steps to predict

    Returns:
    - preds: list of predictions
    - residuals: list of residuals
    """
    preds = []
    residuals = []
    ar_order = len(ar_params)
    ma_order = len(ma_params)

    # Iterate over the data for the given steps
    for i in range(steps):
        # Compute AR and MA terms
        ar_term = sum(ar_params[j] * (data[i - j - 1] if i - j - 1 >= 0 else 0) for j in range(ar_order))
        ma_term = sum(ma_params[j] * (residuals[i - j - 1] if i - j - 1 >= 0 else 0) for j in range(ma_order))

        # One-step prediction
        pred = ar_term + ma_term + constant
        preds.append(pred)

        # Compute residual
        residual = data[i] - pred if i < len(data) else 0
        residuals.append(residual)

    return preds, residuals


#%%
def create_arma_params(ar_coeffs, ma_coeffs):
    # Find the maximum length between ar_coeffs and ma_coeffs
    max_length = max(len(ar_coeffs), len(ma_coeffs))

    # Pad the shorter array with zeros to make their lengths equal
    ar_padded = np.pad(ar_coeffs, (0, max_length - len(ar_coeffs)), 'constant')
    ma_padded = np.pad(ma_coeffs, (0, max_length - len(ma_coeffs)), 'constant')

    # Prepend 1 to both arrays
    ar_params = np.r_[1, ar_padded]
    ma_params = np.r_[1, ma_padded]

    return ar_params, ma_params


# Function for checking stability
def check_stability(ar_params, ma_params):
    ar_roots=abs(np.roots(ar_params))
    ma_roots=abs(np.roots(ma_params))
    if all(abs(root) < 1 for root in ar_roots) and all(abs(root) < 1 for root in ma_roots):
        return True
    else:
        return False

from scipy import signal

def lm_algorithm(na, nb, data):
    # Define initial conditions
    params = np.zeros(na + nb)  # Initial parameters set to zero
    ar_coeffs = params[:na]
    ma_coeffs = params[na:]

    ar_coeffs,ma_coeffs = create_arma_params(ar_coeffs, ma_coeffs)

    # Generate initial system response
    system = signal.dlti(ar_coeffs, ma_coeffs)  # Define discrete-time system
    t, y = signal.dlsim(system, data)  # Simulate system output for initial guess
    error_prev = y.flatten()
    SSE_init=np.dot(error_prev.T, error_prev)
    # print(SSE_init)
    epsilon = 1e-6  # Small perturbation for numerical derivative
    max_iter = 75  # Maximum number of iterations
    lambd = 0.01  # Initial damping factor
    SSE_prev = 10 ** 7
    sse_list = []
    sse_list.append(SSE_init)

    # sse_list.append(np.linalg.norm(error_prev))
    for i in range(max_iter):
        # Compute SSE for current parameters
        # SSE_prev = np.dot(error_prev, error_prev)

        # Initialize Jacobian matrix and error vector
        X = np.zeros((len(data), na + nb))
        for j in range(na + nb):
            # Perturb parameter j by epsilon
            params_new = np.copy(params)
            # print(params_new)
            params_new[j] += epsilon

            # Previous error
            prev_ar,prev_ma =create_arma_params(params[:na],params[na:])

            system_prev = signal.dlti(prev_ar, prev_ma)
            _, error_prev = signal.dlsim(system_prev, data)



            # Perturbed error

            perturbed_ar,perturbed_ma =create_arma_params(params_new[:na],params_new[na:])
            system_perturbed = signal.dlti(perturbed_ar, perturbed_ma)
            _, error_perturbed = signal.dlsim(system_perturbed, data)

            # Calculate perturbed error and numerical derivative
            error_diff = (error_prev - error_perturbed)/epsilon
            error_diff=error_diff.flatten()
            X[:, j] = error_diff

        # print("Reached here")
        # Compute gradient vector and Hessian approximation
        e=error_prev
        g = X.T @ e
        # print(g)
        A = X.T @ X
        # print(A)

        I=np.identity(na + nb)
        # print(I)
        SSE_new=SSE_prev
        # Solve for parameter update
        while SSE_new >= SSE_prev:


            if lambd>=10**20:
                print("Exceeded max value, error ")
                break
            delta = np.linalg.inv(A + lambd* I) @ g
            # print("Delta is ",delta)
            flag=0
            stability_weight=1
            while flag==0:

                params_new = params + stability_weight*delta.flatten()

                # Ensuring the ARMA does not explode

                # params_new = np.clip(params_new, -1, 1)

                # Update system with new parameters
                new_ar,new_ma = create_arma_params(params_new[:na],params_new[na:])
                if check_stability(new_ar,new_ma):
                    flag=1
                else:
                    stability_weight=stability_weight/10





            system_new = signal.dlti(new_ar, new_ma)
            # print("parameters for calculating error",new_ar,new_ma)
            _, y_new = signal.dlsim(system_new, data)

            error_new = y_new.flatten()
            # print("from dlsim last few values",y_new[-10:])
            SSE_new = np.dot(error_new.T, error_new)
            lambd *= 10

        sse_list.append(SSE_new)
        # print(lambd)
        diff=SSE_prev-SSE_new
        params = params_new
        error_prev = error_new
        SSE_prev = SSE_new
        # print("Error values look like",error_new[:10]," and ",error_new[-10:])
        # print("SSE dot product should be",np.dot(error_new.T, error_new))
        # print("New SSE is ",SSE_new)
        # print("Iteration ",i)


        if np.linalg.norm(delta.flatten()) < 10 ** -3:
            break
        else:
            lambd /= 10

    if i>=max_iter:
        print("Error, could not converge")
    else:
        print(f"Succesfully converged in {i} iterations")

    error_var=SSE_new/(len(data)-len(params))
    cov_mat=error_var*np.linalg.inv(A)
    return params, sse_list, cov_mat,error_var


#%%
# Trying higher order on original data for test


ar_order=25
ma_order=24
params, sse_list, cov_mat, error_var = lm_algorithm(ar_order, ma_order, y_train)

#%%
print("Displaying Estimated Coefficients")
ct = 0

for i in range(ar_order):
    print(f"a{i + 1} is {params[ct]:.3f}")
    ct = ct + 1

for i in range(ma_order):
    print(f"b{i + 1} is {params[ct]:.3f}")
    ct = ct + 1

print("Showing confidence intervals")
ct = 0

for i in range(ar_order):
    conf_int = 2 * (cov_mat[ct, ct] ** 0.5)
    print(f"a{i + 1} is between the ranges of {params[ct] - conf_int:.3f} and {params[ct] + conf_int:.3f} with an std of {conf_int:.3f}")
    ct = ct + 1

# Print MA parameters with 3 decimal places
for i in range(ma_order):
    conf_int = 2 * (cov_mat[ct, ct] ** 0.5)
    print(f"b{i + 1} is between the ranges of {params[ct] - conf_int:.3f} and {params[ct] + conf_int:.3f} with an std of {conf_int:.3f}")
    ct = ct + 1

#%%
ar_params=params[:ar_order]
ma_params=params[ar_order:]
print(ar_params)
print(ma_params)

ar_params=[-1*i for i in ar_params]
preds, residuals = one_step_pred(y_train, ar_params, ma_params, len(y_train), 0)

plt.figure(figsize=(12, 6))
plt.plot(y_train[100:200].values, label="Actuals", alpha=0.7)
plt.plot(preds[100:200], label="Predictions", color="red", linestyle="--")
plt.title(f"Predictions vs Actuals 1 step prediction", fontsize=14)
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("Values", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.show()

import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

acf_values = sm.tsa.acf(residuals, fft=True, nlags=30)
print(acf_values[:20])

Q_val = sum(x ** 2 for x in acf_values[1:20])
print("Q val is ", Q_val)
# Plot ACF
plt.figure(figsize=(10, 6))
plt.stem(range(len(acf_values)), acf_values, basefmt=" ")
plt.title("Autocorrelation Function (ACF) of Residuals", fontsize=14)
plt.xlabel("Lag", fontsize=12)
plt.ylabel("ACF Value", fontsize=12)
plt.axhline(y=0, color='black', linewidth=0.8)
plt.axhline(y=1.96 / np.sqrt(len(residuals)), linestyle='--', color='gray', linewidth=1.2,
            label="Significance Threshold")
plt.axhline(y=-1.96 / np.sqrt(len(residuals)), linestyle='--', color='gray', linewidth=1.2)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.show()

ACF_PACF_Plot(residuals, 75)

plt.figure(figsize=(12, 6))
plt.plot(residuals[300:900], label='Residuals', color='purple')
plt.axhline(0, linestyle='--', color='gray', linewidth=1)
plt.xlabel('Index')
plt.ylabel('Residuals')
plt.title('Residuals of SARIMAX Model')
plt.legend()
plt.grid()
plt.show()

# 2. Perform the Ljung-Box test
lb_test_results = acorr_ljungbox(residuals, lags=[10, 20, 30, 50, 100], return_df=True)
print("Ljung-Box Test Results:")
print(lb_test_results)

# Interpretation
for lag, p_value in zip([10, 20, 30, 50, 100], lb_test_results['lb_pvalue']):
    if p_value < 0.05:
        print(f"At lag {lag}, residuals are not white (p-value={p_value:.4f}).")
    else:
        print(f"At lag {lag}, residuals are white (p-value={p_value:.4f}).")

#%%
autocorr_values = sm.tsa.acf(residuals, nlags=200)
gpac_df = gpac_table(autocorr_values, 60, 60)

plt.figure(figsize=(10, 6))
sns.heatmap(gpac_df.iloc[0:11, 20:30], annot=True, fmt=".2f", linewidths=0.5,
            center=0, cbar_kws={'label': 'GPAC Value'})

plt.title(f"GPAC Heatmap for Residuals")
plt.show()

#%%
ar_params_inverted= [-1*i for i in ar_params]

ar_poly = np.append(1, ar_params_inverted)
ma_poly = np.append(1, ma_params)

# Find the roots of the AR and MA polynomials
poles = np.roots(ar_poly)
zeros = np.roots(ma_poly)

# Print the AR and MA polynomials, and their roots
# print("AR Polynomial Coefficients:", ar_poly)
print("AR Polynomial Poles (roots):", poles)
# print("\nMA Polynomial Coefficients:", ma_poly)
print("MA Polynomial Zeros (roots):", zeros)

# Check for zero-pole cancellation
tolerance = 1e-6
cancellation_detected = False

for pole in poles:
    for zero in zeros:
        if np.abs(pole - zero) < tolerance:
            print(f"Zero-Pole Cancellation Detected: Pole {pole} cancels Zero {zero}")
            cancellation_detected = True

# Else condition: If no cancellations are found
if not cancellation_detected:
    print("\nNo zero-pole cancellations detected. The model is already in its simplest version.")

#%%

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
        ma_term = sum(ma_params[j] * (residuals[i - j - 1] if i - j - 1 >= 0 else 0) for j in range(ma_order))

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

steps=100

preds= hstep_forecast(y_train, ar_params, ma_params, steps)

actuals = y_test[:steps].values
plt.figure(figsize=(12, 6))
plt.plot(actuals, label="Actuals", alpha=0.7)
plt.plot(preds, label="Predictions", color="red", linestyle="--")
plt.title("Predictions vs Actuals - h-step Forecast", fontsize=14)
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("Values", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.show()

forecast_errors=actuals-preds
sse = np.sum(forecast_errors**2)
print(f"Sum of Squared Errors (SSE) for forecast: {sse:.2f}")


print(f"Variance of Residuals is {np.var(residuals):.2f}")
print(f"Variance of Forecast errors is {np.var(forecast_errors):.2f}")

print(f"Ratio of Variance of Forecast Errors to Residuals is {np.var(forecast_errors)/np.var(residuals):.2f}")

print(f"Mean of Residuals is {np.mean(residuals):.2f}")
