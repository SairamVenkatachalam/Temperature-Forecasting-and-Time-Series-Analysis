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
# Heatmap
import seaborn as sns

from sklearn.model_selection import train_test_split

train_size = 0.8

train_data, test_data = train_test_split(df, train_size=train_size, shuffle=False)

final_train_data = train_data.drop(columns=['datetime'])
correlation_matrix = final_train_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap")
plt.show()


#%%

# PCA Analysis

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

final_train_data = final_train_data.drop(columns=['T'])
scaler = StandardScaler()

scaled_data = scaler.fit_transform(final_train_data)

# Perform PCA
pca = PCA()
pca.fit(scaled_data)

# Calculate explained variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Plot explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', label='Cumulative Variance')
plt.axhline(y=0.95, color='r', linestyle='dotted', label="95% Explained Variance Threshold")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Explained Variance")
plt.legend()
plt.grid()
plt.show()

# Print the number of components required for 95% explained variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components needed for 95% explained variance: {n_components_95}")

#%%
from numpy.linalg import svd, cond

# Perform SVD
U, S, Vt = svd(scaled_data)

# Calculate condition number
condition_number = cond(scaled_data)
print(f"Condition Number: {condition_number:.2f}")

# Plot singular values
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(S) + 1), S, marker='o', linestyle='-', label='Singular Values')
plt.title("Singular Values from SVD")
plt.xlabel("Index")
plt.ylabel("Singular Value")
plt.grid()
plt.legend()
plt.show()

#%%

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

X = df.drop(columns=['datetime', 'T'])
y = df['T']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on training data
X_test_scaled = scaler.transform(X_test)  # Only transform test data

X_train_scaled_const = sm.add_constant(X_train_scaled)
X_test_scaled_const = sm.add_constant(X_test_scaled)

#%%


# Initialize results DataFrame to store metrics after each feature removal
results_df = pd.DataFrame(columns=['Feature Removed', 'AIC', 'BIC', 'Adjusted R-squared'])


def model_fitting(X, y, prev_feature="None"):
    """Fit the OLS model, perform feature selection and track performance metrics."""

    # Add constant (intercept) to the model
    X_subset = sm.add_constant(X)

    # Fit the OLS model
    model_subset = sm.OLS(y, X_subset).fit()

    # Display summary of the model
    print(model_subset.summary())

    # Extract p-values
    p_values = model_subset.pvalues

    # Remove constant term from p-values
    p_values_without_const = p_values.drop('const')

    # Find feature with the maximum p-value
    max_p_value_feature = p_values_without_const.idxmax()
    max_p_value = p_values_without_const.max()

    # Collect model metrics
    adj_r_squared = model_subset.rsquared_adj
    aic = model_subset.aic
    bic = model_subset.bic

    # Store the results for tracking
    global results_df
    new_row = pd.DataFrame({
        'Feature Removed': [prev_feature],
        'AIC': [aic],
        'BIC': [bic],
        'Adjusted R-squared': [adj_r_squared]
    })

    # Append new row to results DataFrame
    results_df = pd.concat([results_df, new_row], ignore_index=True)

    print(f"\nFeature with the highest p-value (least significance): {max_p_value_feature}")
    print(f"P-value: {max_p_value:.2f}\n")

    return max_p_value_feature


X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)


X_subset = X_train_df.copy()


max_p_value_feature = model_fitting(X_subset, y_train)
print(results_df)

#%%
X_subset = X_subset.drop(columns=[max_p_value_feature])
max_p_value_feature=model_fitting(X_subset, y_train,max_p_value_feature)
print(results_df)

#%%
X_subset = X_subset.drop(columns=[max_p_value_feature])
max_p_value_feature=model_fitting(X_subset, y_train,max_p_value_feature)
print(results_df)

#%%
X_subset = X_subset.drop(columns=[max_p_value_feature])
max_p_value_feature=model_fitting(X_subset, y_train,max_p_value_feature)
print(results_df)


#%%
X_subset = X_subset.drop(columns=[max_p_value_feature])
max_p_value_feature=model_fitting(X_subset, y_train,max_p_value_feature)
print(results_df)

#%%
X_subset = X_subset.drop(columns=[max_p_value_feature])
max_p_value_feature=model_fitting(X_subset, y_train,max_p_value_feature)
print(results_df)

#%%
X_subset = X_subset.drop(columns=[max_p_value_feature])
max_p_value_feature=model_fitting(X_subset, y_train,max_p_value_feature)
print(results_df)

#%%
X_subset = X_subset.drop(columns=[max_p_value_feature])
max_p_value_feature=model_fitting(X_subset, y_train,max_p_value_feature)
print(results_df)

#%%
X_subset = X_subset.drop(columns=[max_p_value_feature])
max_p_value_feature=model_fitting(X_subset, y_train,max_p_value_feature)
print(results_df)

#%%
X_subset = X_subset.drop(columns=[max_p_value_feature])
max_p_value_feature=model_fitting(X_subset, y_train,max_p_value_feature)
print(results_df)
#%%
X_subset = X_subset.drop(columns=[max_p_value_feature])
max_p_value_feature=model_fitting(X_subset, y_train,max_p_value_feature)
print(results_df)
#%%
initial_features = X_train_df.columns.tolist()

removed_features = [column for column in X_train_df if column not in X_subset]

remaining_features = [feature for feature in initial_features if feature not in removed_features]

print(f"Based on the backward stepwise regression, we should keep {len(remaining_features)} features: {remaining_features}, "
      f"and remove {len(removed_features)} features: {removed_features}.")

#%%
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


X_train_selected = X_train[['RH', 'AH']]
X_train_selected_const = sm.add_constant(X_train_selected)

X_test_selected = X_test[['RH', 'AH']]
X_test_selected_const = sm.add_constant(X_test_selected)


# Fit the model with constant included
model = sm.OLS(y_train, X_train_selected_const).fit()

# Display the model summary
print(model.summary())
y_pred = model.predict(X_test_selected_const)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
aic = model.aic
bic = model.bic
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test_selected.shape[1] - 1)

# Display the metrics
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"AIC: {aic:.4f}")
print(f"BIC: {bic:.4f}")
print(f"R-squared: {r2:.4f}")
print(f"Adjusted R-squared: {adj_r2:.4f}")
# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:100], label="Actual")
plt.plot(y_pred.values[:100], label="Predicted", linestyle='--')
plt.legend()
plt.title('One-Step Ahead Prediction')
plt.show()

#%%
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Select the final features (RH and AH)
X_final = X_train[['RH', 'AH']]

# Add constant (intercept) to the final features
X_final_const = sm.add_constant(X_final)
vif_data = pd.DataFrame()
vif_data["Feature"] = X_final.columns
vif_data["VIF"] = [variance_inflation_factor(X_final.values, i) for i in range(X_final.shape[1])]

# Display the VIF results
print(vif_data)

#%%
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.model_selection import TimeSeriesSplit

# Prepare the training and testing datasets
X_train_selected = X_train[['RH', 'AH']]
X_train_selected_const = sm.add_constant(X_train_selected)

X_test_selected = X_test[['RH', 'AH']]
X_test_selected_const = sm.add_constant(X_test_selected)

# Fit the model
model = sm.OLS(y_train, X_train_selected_const).fit()

# Display the model summary
print(model.summary())

# Predictions
y_pred = model.predict(X_test_selected_const)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
aic = model.aic
bic = model.bic
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test_selected.shape[1] - 1)

# Display the metrics
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"AIC: {aic:.4f}")
print(f"BIC: {bic:.4f}")
print(f"R-squared: {r2:.4f}")
print(f"Adjusted R-squared: {adj_r2:.4f}")

# Hypothesis Tests
# F-test: Overall significance of the model
residuals = y_train - model.fittedvalues
print(f"F-statistic: {model.fvalue:.4f}, p-value: {model.f_pvalue:.4f}")
print(f"Mean of residuals: {residuals.mean():.4f}")
print(f"Variance of residuals: {residuals.var():.4f}")

for i, p in enumerate(model.pvalues):
    print(f"t-statistic for {X_train_selected_const.columns[i]}: {model.tvalues[i]:.4f}, p-value: {p:.4f}")

#%%
# Importing required libraries for cross-validation
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Initialize TimeSeriesSplit with the desired number of splits
tscv = TimeSeriesSplit(n_splits=5)

# Prepare to store cross-validation results
cv_results = {
    'fold': [],
    'mse': [],
    'rmse': [],
    'r2': [],
    'adj_r2': []
}

# Cross-validation loop
for fold, (train_index, test_index) in enumerate(tscv.split(X_train_selected_const)):
    # Split the data into training and testing sets for this fold
    X_train_cv, X_test_cv = X_train_selected_const.iloc[train_index], X_train_selected_const.iloc[test_index]
    y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]

    # Fit the model
    model_cv = sm.OLS(y_train_cv, X_train_cv).fit()

    # Predictions
    y_pred_cv = model_cv.predict(X_test_cv)

    # Calculate metrics
    mse_cv = mean_squared_error(y_test_cv, y_pred_cv)
    rmse_cv = np.sqrt(mse_cv)
    r2_cv = r2_score(y_test_cv, y_pred_cv)
    adj_r2_cv = 1 - (1 - r2_cv) * (len(y_test_cv) - 1) / (len(y_test_cv) - X_test_cv.shape[1] - 1)

    # Store results
    cv_results['fold'].append(fold + 1)
    cv_results['mse'].append(mse_cv)
    cv_results['rmse'].append(rmse_cv)
    cv_results['r2'].append(r2_cv)
    cv_results['adj_r2'].append(adj_r2_cv)

# Display cross-validation results
import pandas as pd

cv_results_df = pd.DataFrame(cv_results)
print("Cross-validation results:")
print(cv_results_df)

# Summary of cross-validation performance
print("\nSummary:")
print(f"Mean MSE: {cv_results_df['mse'].mean():.4f}, Std Dev: {cv_results_df['mse'].std():.4f}")
print(f"Mean RMSE: {cv_results_df['rmse'].mean():.4f}, Std Dev: {cv_results_df['rmse'].std():.4f}")
print(f"Mean R-squared: {cv_results_df['r2'].mean():.4f}, Std Dev: {cv_results_df['r2'].std():.4f}")
print(f"Mean Adjusted R-squared: {cv_results_df['adj_r2'].mean():.4f}, Std Dev: {cv_results_df['adj_r2'].std():.4f}")

# Optional: Visualize the metrics
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(cv_results['fold'], cv_results['mse'], label='MSE', marker='o')
plt.plot(cv_results['fold'], cv_results['rmse'], label='RMSE', marker='o')
plt.plot(cv_results['fold'], cv_results['r2'], label='R-squared', marker='o')
plt.plot(cv_results['fold'], cv_results['adj_r2'], label='Adjusted R-squared', marker='o')
plt.xlabel('Fold')
plt.ylabel('Metric Value')
plt.title('Cross-Validation Metrics')
plt.legend()
plt.grid(True)
plt.show()

#%%
def ACF_plot(y, lags, equation=0):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)

    fig = plt.figure()
    lags_range = range(len(acf))

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
    plt.show()



# Predictions on train set
y_pred = model.predict(X_train_selected_const)

residuals=y_train-y_pred
ACF_plot(residuals,50)

from statsmodels.stats.diagnostic import acorr_ljungbox
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
lb_test_results = acorr_ljungbox(residuals, lags=[20], return_df=True)

#%%
print(f'The mean of residuals are {np.mean(residuals):0.2f}')

