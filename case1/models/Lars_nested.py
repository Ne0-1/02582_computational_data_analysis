import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from functions import normalize, centerData
from sklearn.model_selection import KFold
from sklearn.linear_model import Lars

# Sources for approach
# https://stats.stackexchange.com/questions/254612/how-to-obtain-optimal-hyperparameters-after-nested-cross-validation
# Summary of link: nested cross-validation is to estimate predictive performance / generalization error

def setup_mpl():
    mpl.rcParams['font.family'] = 'Times New Roman'
    return None

# Reading response and independent variablescase1/data/response.txt
y = np.loadtxt('case1/data/response.txt')
X = np.loadtxt('case1/data/independent.txt')


(n, p) = X.shape

# Define outer cross validation split
CV_outer = 10
CV_inner = 5
kf1 = KFold(n_splits=CV_outer, random_state=42, shuffle=True)

Err_par = np.zeros(CV_outer)
Err_test = np.zeros(CV_outer)
p_opts = np.zeros(CV_outer)
# From 02450 lecture notes (page 175)
# Outer cross-validation loop - Estimate generalization error
for i, (outer_train_index, outer_test_index) in enumerate(kf1.split(X)):

    y_par = y[outer_train_index]
    X_par = X[outer_train_index, :]

    y_test = y[outer_test_index]
    X_test = X[outer_test_index, :]

    # Define inner cross validation split
    kf2 = KFold(n_splits=CV_inner, random_state=42, shuffle=True)
    
    # Define hyper parameter space
    stop = n-math.ceil(n/CV_inner)
    K = range(stop)

    # Prepare memory storage
    Err_tr = np.zeros((CV_inner, len(K)))
    Err_val = np.zeros((CV_inner, len(K)))

    # Inner cross-validation loop - select optimal model
    for j, (inner_train_index, inner_test_index) in enumerate(kf2.split(X_par)):
            
        X_train = X_par[inner_train_index, :]
        y_train = y_par[inner_train_index]

        X_val = X_par[inner_test_index, :]
        y_val = y_par[inner_test_index]

        # Imputing missing values
        X_train = np.where(np.isnan(X_train), np.ma.array(X_train, mask=np.isnan(X_train)).mean(axis=0), X_train)
        X_val = np.where(np.isnan(X_val), np.ma.array(X_val, mask=np.isnan(X_val)).mean(axis=0), X_val)
            
        # Center data, notice test is centered using training mean
        y_train, mu = centerData(y_train)
        y_val = y_val - mu

        X_train, mu = centerData(X_train)
        X_val = X_val - mu

        # Normalize val using train
        X_train, d = normalize(X_train)
        X_val = X_val / d

        for p in K:
            model_inner = Lars(n_nonzero_coefs=p, fit_path = False, fit_intercept = False)
            model_inner.fit(X_train, y_train)
            
            # Training predictions
            beta_inner = model_inner.coef_.ravel()
            y_hat_train = X_train @ beta_inner  # Don't need intercept due to centered data
            y_hat_val = X_val @ beta_inner      # Don't need intercept due to centered data

            # Compute RMSE for train and validation set
            Err_tr[j, p] = np.sqrt(((y_train - y_hat_train)**2).mean())
            Err_val[j, p] = np.sqrt(((y_val - y_hat_val)**2).mean())

    # Select optimal hyperparameter
    p_opt = np.argmin(Err_val.mean(axis=0))

    # Imputing missing values
    X_par = np.where(np.isnan(X_par), np.ma.array(X_par, mask=np.isnan(X_par)).mean(axis=0), X_par)
    X_test = np.where(np.isnan(X_test), np.ma.array(X_test, mask=np.isnan(X_test)).mean(axis=0), X_test)

    # Center data, notice test is centered using training mean
    y_par, mu = centerData(y_par)
    y_test = y_test - mu

    X_par, mu = centerData(X_par)
    X_test = X_test - mu

    # Normalize val using train
    X_par, d = normalize(X_par)
    X_test = X_test / d

    # Train optimal model
    model_outer = Lars(n_nonzero_coefs=p_opt, fit_path = False, fit_intercept = False)
    model_outer.fit(X_par, y_par)
    
    # Training predictions
    beta_outer = model_outer.coef_.ravel()
    y_hat_par =  X_par @ beta_outer  # Adding E[y_par] because model is fitted without intercept
    y_hat_test =  X_test @ beta_outer  # Adding E[y_par] because model is fitted without intercept

    # Compute test error
    Err_par[i] = np.sqrt(((y_par - y_hat_par)**2).mean())
    Err_test[i] = np.sqrt(((y_test - y_hat_test)**2).mean())
    p_opts[i] = p_opt

# Compute estimate of generalization error
E_gen = Err_test.mean()

# Results:
print('Computed estimated test error E_test[i]: ', Err_test)
print('')
print('Correspondind optimal hyperparamter p*: ', p_opts)
print('')
print('Estimated generalization error: ')
print(E_gen)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=100)

ax1.plot(K, Err_tr.T, ':', alpha=0.5)
ax1.plot(Err_tr.mean(axis=0), c='k', label='Average training error', linewidth=1.5)
ax1.set_title('Training error for each fold')
ax1.legend()
ax1.set_xlabel(r"non-zero $\beta$")
ax1.set_ylabel('RMSE')

ax2.plot(K, Err_val.T, ':', alpha=0.5)
ax2.plot(Err_val.mean(axis=0), c='k', label='Average training error', linewidth=1.5)
#ax2.axvline(p_opt, label=r"$p^{*}$", linestyle='dashed', c='k', alpha=0.75)
#[ax.axvline(_x, linewidth=1, color='k', linestyle='dashed', alpha=0.75) for _x in p_opts]
ax2.set_title('Validation error for each fold')
ax2.set_yscale('log')
ax2.legend()
ax2.set_xlabel(r"non-zero $\beta$")
ax2.set_ylabel('RMSE')
plt.show()

fig, ax = plt.subplots(figsize=(15,5), dpi=100)
ax.plot(K, Err_tr.mean(axis=0), c='darkblue', label='Average training error')
ax.plot(K, Err_val.mean(axis=0), c='firebrick', label='Average validation error')
#ax.axvline(p_opt, label=r"$p^{*}$", linestyle='dashed', c='k', alpha=0.75)
[ax.axvline(_x, linewidth=1, color='k', linestyle='dashed', alpha=0.75) for _x in p_opts]
ax.set_yscale('log')
ax.legend()
ax.set_xlabel(r"non-zero $\beta$")
ax.set_ylabel('RMSE')
plt.show()
