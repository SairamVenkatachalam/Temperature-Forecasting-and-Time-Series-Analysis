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
import seaborn as sns

from sklearn.model_selection import train_test_split


#%%
import numpy as np
def autocorrelation(data,lags):

    autocorrelation=np.zeros(lags+1)
    mean=np.mean(data)
    variance=np.var(data)
    # print(mean)
    for i in range(0,lags+1):
        corr=0
        for j in range(i,len(data)):
            corr=corr+(data[j]-mean)*(data[j-i]-mean)/((variance)*len(data))

        autocorrelation[i]=corr

    autocorr_full = np.concatenate((autocorrelation[:0:-1], autocorrelation))  # Mirror positive lags to negative lags

    lags_range = np.arange(-lags, lags + 1)

    # Create the stem plot
    markerline, stemlines, baseline = plt.stem(lags_range, autocorr_full, linefmt='b-', basefmt=' ', markerfmt='ro')

    # Set red color for all dots
    markerline.set_markerfacecolor('red')

    n = len(data)
    # Plot significance bands
    significance_level = 1.96 / np.sqrt(n)
    plt.fill_between(lags_range, -significance_level, significance_level, color='purple', alpha=0.2,
                     label='Insignificance Region')

    plt.title('Autocorrelation Function of Data')
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelation')
    plt.grid(True)
    plt.legend()
    plt.show()
    return

def cal_autocov(data,lags):

    autocorrelation=np.zeros(lags+1)

    for i in range(0,lags+1):
        corr=0
        for j in range(i,len(data)-lags):
            corr=corr+data[j]*data[j+i]

        autocorrelation[i]=corr/(len(data)-lags)

    return autocorrelation

def cal_crosscov(x,y,lags):

    crosscorrelation=np.zeros(lags+1)

    for i in range(0,lags+1):
        corr=0
        for j in range(i,len(x)-lags):
            corr=corr+x[j]*y[j+i]

        crosscorrelation[i]=corr/(len(x)-lags)

    return crosscorrelation




def check_stationarity(data):
    print(f"ADF Test for the data")
    ts.ADF_Cal(data)
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

def exp_autocorrelation(u, tau_max):
    """
    Calculate experimental autocorrelation R_u(tau) efficiently.

    Parameters:
        u (array-like): Input signal.
        tau_max (int): Maximum lag (K in the formula).

    Returns:
        array: Autocorrelation values for tau = 0 to tau_max.
    """
    N = len(u)
    autocorr = np.zeros(tau_max + 1)

    for tau in range(tau_max + 1):
        autocorr[tau] = np.dot(u[:N - tau], u[tau:N]) / (N - tau)

    return autocorr


def exp_cross_correlation(u, y, tau_max):
    """
    Calculate experimental cross-correlation R_uy(tau) efficiently.

    Parameters:
        u (array-like): Input signal u.
        y (array-like): Output signal y.
        tau_max (int): Maximum lag (K in the formula).

    Returns:
        array: Cross-correlation values for tau = 0 to tau_max.
    """
    N = len(u)
    cross_corr = np.zeros(tau_max + 1)

    for tau in range(tau_max + 1):
        cross_corr[tau] = np.dot(u[:N - tau], y[tau:N]) / (N - tau)

    return cross_corr

def construct_matrix(autocorr, tau_max):
    """
    Construct the matrix R_u(tau) based on autocorrelation.

    Parameters:
        autocorr (array): Autocorrelation values (from exp_autocorrelation).
        tau_max (int): Maximum lag (K in the formula).

    Returns:
        2D array: R_u(tau) matrix of size (K+1, K+1).
    """
    matrix_size = tau_max + 1
    R_u_matrix = np.zeros((matrix_size, matrix_size))

    for i in range(matrix_size):
        for j in range(matrix_size):
            lag = abs(i - j)
            R_u_matrix[i, j] = autocorr[lag] if lag < len(autocorr) else 0
    return R_u_matrix

def construct_array(autocorr, tau_max):

    array_size = tau_max + 1
    R_uy_array = np.zeros(array_size)

    for i in range(array_size):
            R_uy_array[i] = autocorr[i]
    return R_uy_array








#%%
y=df['T']
train_size = int(0.8 * len(y))
y_train = y[:train_size]
y_test = y[train_size:]


u=df['AH']
train_size = int(0.8 * len(u))
u_train = u[:train_size]
u_test = u[train_size:]

K = 50

autocorr = exp_autocorrelation(u_train, K)

cross_corr = exp_cross_correlation(u_train, y_train, K)

R_u_matrix = construct_matrix(autocorr, K)

R_uy_array = construct_array(cross_corr,K)
print("R_u(tau) Matrix:")
print(R_u_matrix[:5, :5])

print("R_u_y_array is ")
print(R_uy_array[:5])

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

            tolerance = 10 ** -10

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
def compute_g_k(R_u_matrix, R_uy_array):
    # Inverse of R_u_matrix
    R_u_inv = np.linalg.inv(R_u_matrix)
    # Compute g_hat(k)
    g_k = np.dot(R_u_inv, R_uy_array)
    return g_k

def estimate_v(t_series, u_series, g_k, K):
    """
    Estimate v_hat(t) using the given formula.
    Args:
    - t_series (array): Observed output y(t).
    - u_series (array): Input signal u(t).
    - g_k (array): Coefficients g_hat(i).
    - K (int): Maximum lag K.

    Returns:
    - v_hat (array): Estimated residual signal v_hat(t).
    """
    v_hat = []
    N = len(t_series)  # Length of the signal

    # Compute v_hat for each time step t
    for t in range(N):
        sum_term = 0
        for i in range(K + 1):
            if t - i >= 0:  # Ensure index is valid
                sum_term += g_k[i] * u_series[t - i]
        v_hat_t = t_series[t] - sum_term
        v_hat.append(v_hat_t)

    return np.array(v_hat)

#%%
import statsmodels.api as sm

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


def CCF_plot(errors, alpha_t, lags):
    """
    Plot Cross Correlation Function (CCF) of residuals and explanatory variables.

    Parameters:
    - errors: Residuals (numpy array or similar)
    - alpha_t: Explanatory variable (numpy array or similar)
    - lags: Number of lags to compute the cross-correlation for
    """
    ccf = sm.tsa.stattools.ccf(errors, alpha_t)[:lags + 1]  # Compute cross-correlation for given lags
    fig = plt.figure(figsize=(10, 6))
    lags_range = range(len(ccf))

    plt.title('Cross Correlation of Residuals', fontsize=14)
    markerline, stemlines, baseline = plt.stem(lags_range, ccf, linefmt='b-', basefmt=' ', markerfmt='ro')
    markerline.set_markerfacecolor('red')

    # Significance level (95% confidence)
    n = len(errors)
    significance_level = 1.96 / np.sqrt(n)
    plt.fill_between(lags_range, -significance_level, significance_level, color='purple', alpha=0.2,
                     label='Insignificance Region')

    plt.xlabel('Lags', fontsize=12)
    plt.ylabel('Cross Correlation', fontsize=12)
    plt.ylim(-1, 1)  # Set y-axis limits
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
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

season=24
y_train_diff =seasonal_difference(y_train,season)

# Further differencing non seasonal
season=1
y_train_diff =seasonal_difference(y_train_diff,season)



season=24
u_train_diff =seasonal_difference(u_train,season)

# Further differencing non seasonal
season=1
u_train_diff =seasonal_difference(u_train_diff,season)


#%%
CCF_plot(y_train_diff,u_train_diff,70)

def estimate_v(t_series, u_series, g_k, K):
    """
    Estimate v_hat(t) using the given formula.
    Args:
    - t_series (array): Observed output y(t).
    - u_series (array): Input signal u(t).
    - g_k (array): Coefficients g_hat(i).
    - K (int): Maximum lag K.

    Returns:
    - v_hat (array): Estimated residual signal v_hat(t).
    """
    v_hat = []
    N = len(t_series)  # Length of the signal

    # Compute v_hat for each time step t
    for t in range(N):
        sum_term = 0
        for i in range(K + 1):
            if t - i >= 0:  # Ensure index is valid
                sum_term += g_k[i] * u_series[t - i]
        v_hat_t = t_series[t] - sum_term
        v_hat.append(v_hat_t)

    return np.array(v_hat)
#%%
K = 100

autocorr = exp_autocorrelation(u_train_diff, K)

cross_corr = exp_cross_correlation(u_train_diff, y_train_diff, K)

R_u_matrix = construct_matrix(autocorr, K)

R_uy_array = construct_array(cross_corr,K)

g_k = compute_g_k(R_u_matrix, R_uy_array)

CCF_plot(y_train_diff,u_train_diff,50)

ACF_PACF_Plot(y_train_diff,50)

gpac_df = gpac_table(g_k, 50,50)


#%%
plt.figure(figsize=(10, 6))
sns.heatmap(gpac_df.iloc[0:10, 0:10], annot=True, fmt=".2f", linewidths=0.5,
            center=0, cbar_kws={'label': 'GPAC Value'})

plt.title(f"G-GPAC table")
plt.show()

#%%
v_hat = estimate_v(y_train, u_train, g_k, K)

autocorr=exp_autocorrelation(v_hat,K)

gpac_df = gpac_table(autocorr, 30,30)

#%%

plt.figure(figsize=(10, 6))
sns.heatmap(gpac_df.iloc[20:30, 0:13], annot=True, fmt=".2f", linewidths=0.5,
            center=0, cbar_kws={'label': 'GPAC Value'})

plt.title(f"H-GPAC table")
plt.show()

#%%
from scipy import signal
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
def check_stability(ar_params, ma_params):
    ar_roots=abs(np.roots(ar_params))
    ma_roots=abs(np.roots(ma_params))
    if all(abs(root) < 1 for root in ar_roots) and all(abs(root) < 1 for root in ma_roots):
        return True
    else:
        return False


def lm_algorithm_box(nb, nf,nc,nd,data,u):
    # Define initial conditions
    params = np.zeros(nb + nf+nc+nd+1)  # Initial parameters set to zero
    params[0]=1
    b=params[1:nb+1]
    f = params[nb + 1:nb+1+nf]
    c = params[nb+1+nf:nb+1+nf+nc]
    d = params[nb+1+nf+nc:nb+1+nf+nc+nd]

    b,f = create_arma_params(b, f)
    b[0]=params[0]
    c,d = create_arma_params(c, d)


    # Generate initial system response
    system = signal.dlti(d,c)  # Define discrete-time system
    t, y1 = signal.dlsim(system, data)  # Simulate system output for initial guess
    # print(d)
    # print(c)
    system = signal.dlti(np.convolve(b,d), np.convolve(f,c))  # Define discrete-time system
    t, y2 = signal.dlsim(system, u)  # Simulate system output for initial guess
    # print(np.convolve(b,d))
    # print(np.convolve(f,c))
    error_prev = y1.flatten()-y2.flatten()


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

        X = np.zeros((len(data), len(params)))
        for j in range(len(params)):
            # Perturb parameter j by epsilon
            params_new = np.copy(params)
            # print(params_new)
            params_new[j] += epsilon



            # Perturbed error

            b = params_new[1:nb + 1]
            f = params_new[nb + 1:nb + 1 + nf]
            c = params_new[nb + 1 + nf:nb + 1 + nf + nc]
            d = params_new[nb + 1 + nf + nc:nb + 1 + nf + nc + nd]

            b, f = create_arma_params(b, f)
            b[0] = params_new[0]
            c, d = create_arma_params(c, d)

            # Generate initial system response
            system = signal.dlti(d, c)  # Define discrete-time system
            t, y1 = signal.dlsim(system, data)  # Simulate system output for initial guess

            system = signal.dlti(np.convolve(b, d), np.convolve(f, c))  # Define discrete-time system
            t, y2 = signal.dlsim(system, u)  # Simulate system output for initial guess

            error_perturbed = y1.flatten() - y2.flatten()

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

        I=np.identity(len(params))
        # print(I)
        SSE_new=SSE_prev
        print(f"Iteration : {i} - SSE - {SSE_new}")
        # Solve for parameter update
        while SSE_new >= SSE_prev:


            if lambd>=10**20:
                print("Exceeded max value, error ")
                break
            delta = np.linalg.inv(A + lambd* I) @ g
            # print("Delta is ",delta)
            # Removing stabillity check
            flag=0
            stability_weight=1
            while flag==0:

                params_new = params + stability_weight*delta.flatten()

                # Ensuring the ARMA does not explode

                # params_new = np.clip(params_new, -1, 1)

                # Update system with new parameters
                new_f,new_d = create_arma_params(params_new[nb + 1:nb + 1 + nf],params_new[nb + 1 + nf + nc:nb + 1 + nf + nc + nd])

                # print(new_f)
                # print(new_d)
                if check_stability(new_f,new_d):
                    flag=1
                else:
                    stability_weight=stability_weight/10





            # Perturbed error

            b = params_new[1:nb + 1]
            f = params_new[nb + 1:nb + 1 + nf]
            c = params_new[nb + 1 + nf:nb + 1 + nf + nc]
            d = params_new[nb + 1 + nf + nc:nb + 1 + nf + nc + nd]

            b, f = create_arma_params(b, f)
            b[0] = params_new[0]
            c, d = create_arma_params(c, d)

            # Generate initial system response
            system = signal.dlti(d, c)  # Define discrete-time system
            t, y1 = signal.dlsim(system, data)  # Simulate system output for initial guess

            system = signal.dlti(np.convolve(b, d), np.convolve(f, c))  # Define discrete-time system
            t, y2 = signal.dlsim(system, u)  # Simulate system output for initial guess

            error_new = y1.flatten() - y2.flatten()
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
        # print(params)

    error_var=SSE_new/(len(data)-len(params))
    # print(A)
    # cov_mat=error_var*np.linalg.inv(A)
    # Using pseudo inverse because of singularity
    cov_mat = error_var * np.linalg.pinv(A)
    # cov_mat=[]
    return params, sse_list, cov_mat,error_var
#%%
# Best params so far
# nb=0
# nf=27
# nc=1
# nd=32


nb=28
nf=31
nc=24
nd=33


# params,sse_list,cov_mat,error_var=lm_algorithm_box(1,1,1,1,y,u)
params,sse_list,cov_mat,error_var=lm_algorithm_box(nb,nf,nc,nd,y_train,u_train)
#%%
print("Displaying Estimated Coefficients")
ct=0

for i in range(nb+1):
    print(f"b{i} is {params[ct]:.3f}")
    ct=ct+1

for i in range(nf):

    print(f"f{i} is {params[ct]:.3f}")
    ct=ct+1

for i in range(nc):

    print(f"c{i} is {params[ct]:.3f}")
    ct=ct+1

for i in range(nd):

    print(f"d{i} is {params[ct]:.3f}")
    ct=ct+1

print("Showing confidence intervals")
ct=0

for i in range(nb+1):
    conf_int = 2 * (cov_mat[ct, ct] ** 0.5)
    print(f"b{i} is between the ranges of {params[ct] - conf_int:.3f} and {params[ct] + conf_int:.3f} with a std of {conf_int:.3f}")
    ct = ct + 1

for i in range(nf):
    conf_int = 2 * (cov_mat[ct, ct] ** 0.5)
    print(f"f{i+1} is between the ranges of {params[ct] - conf_int:.3f} and {params[ct] + conf_int:.3f} with a std of {conf_int:.3f}")
    ct = ct + 1

for i in range(nc):
    conf_int = 2 * (cov_mat[ct, ct] ** 0.5)
    print(f"c{i+1} is between the ranges of {params[ct] - conf_int:.3f} and {params[ct] + conf_int:.3f} with a std of {conf_int:.3f}")
    ct = ct + 1

for i in range(nd):
    conf_int = 2 * (cov_mat[ct, ct] ** 0.5)
    print(f"d{i+1} is between the ranges of {params[ct] - conf_int:.3f} and {params[ct] + conf_int:.3f} with a std of {conf_int:.3f}")
    ct = ct + 1

#%%
b = params[:nb + 1]
f = params[nb + 1:nb + 1 + nf]
c = params[nb + 1 + nf:nb + 1 + nf + nc]
d = params[nb + 1 + nf + nc:nb + 1 + nf + nc + nd]


#%%
from scipy.stats import chi2


def one_step_forecast(b,f,c,d,u,y,steps):

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

#%%


def whiteness_test(errors, M, alpha=0.05):
    N = len(errors)
    acf = np.correlate(errors, errors, mode='full') / (errors.var() * N)
    acf = acf[len(acf) // 2:]  # Keep positive lags only

    Q = N * np.sum((acf[1:M + 1]) ** 2)  # Exclude lag 0
    Q_star = chi2.ppf(1 - alpha, M)

    return Q, Q_star


def s_statistics(errors, alpha_t, M, alpha=0.05):
    N = len(errors)
    cross_corr = np.correlate(errors, alpha_t, mode='full') / (np.sqrt(errors.var() * alpha_t.var()) * N)
    cross_corr = cross_corr[len(cross_corr) // 2:]  # Keep positive lags only

    S = N * np.sum((cross_corr[:M + 1]) ** 2)
    S_star = chi2.ppf(1 - alpha, M)

    return S, S_star

#%%

steps=len(y_train)
preds,residuals = one_step_forecast(b,f,c,d,u,y_train,steps)
errors = y[:steps] - preds

plt.figure(figsize=(12, 6))
plt.plot(y_train[400:500].values, label="Actuals", alpha=0.7)
plt.plot(preds[400:500], label="Predictions", color="red", linestyle="--")
plt.title(f"Box Jenkins Model Predictions vs Actuals 1 step prediction", fontsize=14)
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("Values", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.show()

#%%
ACF_PACF_Plot(errors, 20)

#%%
# Q test
Q, Q_star = whiteness_test(errors, 20)
print(f"Q-statistics: {Q}, Q*: {Q_star}")

if Q<Q_star:
    print("Q test passed! The residuals are White !")
else:
    print("Q test failed. The residuals are colored")



#%%
# Prewhitening of Input

autocorr_values = sm.tsa.acf(u_train_diff,nlags=200)
gpac_df = gpac_table(autocorr_values, 60,60)



plt.figure(figsize=(10, 6))
sns.heatmap(gpac_df.iloc[20:30, 0:10], annot=True, fmt=".2f", linewidths=0.5,
            center=0, cbar_kws={'label': 'GPAC Value'})

plt.title(f"GPAC Heatmap for Absolute Humidity (Stationary)")
plt.show()

#%%
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
# PRe whitening u to get the S test

ar_order=31
ma_order=24
params, sse_list, cov_mat, error_var = lm_algorithm(ar_order, ma_order, u_train)

#%%
ar_params=params[:ar_order]
ma_params=params[ar_order:]
print(ar_params)
print(ma_params)

def pre_whiten(u_train,ar_params,ma_params):
    # error=[]
    ar_params,ma_params=create_arma_params(ar_params,ma_params)
    print(ar_params)
    print(ma_params)
    system = signal.dlti(ar_params, ma_params)
    _, error = signal.dlsim(system, u_train)
    return error

#%%
alpha=pre_whiten(u_train,ar_params,ma_params)
ACF_PACF_Plot(alpha,30)
alpha=alpha.flatten()


#%%
# S test
S, S_star = s_statistics(errors, alpha, 20)
print(f"S-statistics: {S}, S*: {S_star}")

if S<S_star:
    print("S test passed! The cross correlation is White !")
else:
    print("S test failed. The cross correlation is colored")

#%%

CCF_plot(errors,alpha,20)

#%%
def h_step_forecast(b, f, c, d, u, y, steps):
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
    one_step_preds, one_step_residuals = one_step_forecast(b, f, c, d, u, y_train, len(y_train))
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
steps = 400
start_step = len(y_train)
preds = h_step_forecast(b, f, c, d, u, y_train, steps)

# Visualization
actuals = y[start_step:start_step + steps].values

forecast_errors=actuals-preds
plt.figure(figsize=(12, 6))
plt.plot(actuals, label="Actuals", alpha=0.7)
plt.plot(preds, label="Predictions", color="red", linestyle="--")
plt.title("Box Jenkins Model - h-step Forecast", fontsize=14)
plt.xlabel("Time Steps", fontsize=12)
plt.ylabel("Values", fontsize=12)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.show()

sse = np.sum(forecast_errors**2)
print(f"Sum of Squared Errors (SSE) for forecast: {sse:.2f}")


print(f"Variance of Residuals is {np.var(residuals):.2f}")
print(f"Variance of Forecast errors is {np.var(forecast_errors):.2f}")

print(f"Ratio of Variance of Forecast Errors to Residuals is {np.var(forecast_errors)/np.var(residuals):.2f}")
print(f"Mean of Residuals is {np.mean(residuals):.2f}")


#%%
ar_poly = np.append(1, f)
ma_poly = np.append(1, b)

# Find the roots of the AR and MA polynomials
poles = np.roots(ar_poly)
zeros = np.roots(ma_poly)

# Print the AR and MA polynomials, and their roots
# print("AR Polynomial Coefficients:", ar_poly)
print("G Polynomial Poles (roots):", poles)
# print("\nMA Polynomial Coefficients:", ma_poly)
print("G Polynomial Zeros (roots):", zeros)

# Check for zero-pole cancellation
tolerance = 1e-6
cancellation_detected = False

for pole in poles:
    for zero in zeros:
        if np.abs(pole - zero) < tolerance:
            print(f"Zero-Pole Cancellation Detected: Pole {pole} cancels Zero {zero}")
            cancellation_detected = True

# Checking zero pole cancellation for c and d

ar_poly = np.append(1, d)
ma_poly = np.append(1, c)

# Find the roots of the AR and MA polynomials
poles = np.roots(ar_poly)
zeros = np.roots(ma_poly)

# Print the AR and MA polynomials, and their roots
# print("AR Polynomial Coefficients:", ar_poly)
print("H Polynomial Poles (roots):", poles)
# print("\nMA Polynomial Coefficients:", ma_poly)
print("H Polynomial Zeros (roots):", zeros)

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
# This code was used just to store params, need not be executed everytime as the LM usually converges to similar values each time
#
# np.save('Best_Model_final3_b_original.npy', b)
# np.save('Best_Model_final3_f_original.npy', f)
#
# np.save('Best_Model_final3_c_original.npy', c)
# np.save('Best_Model_final3_d_original.npy', d)

# print("Arrays saved to .npy files.")
