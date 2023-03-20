import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from functions import normalize, centerData
from sklearn.model_selection import KFold
from sklearn.linear_model import Lars

#TODO: Maybe change to nested cross-validation
#TODO: All methods should be evaluated on the same splits!

def setup_mpl():
    mpl.rcParams['font.family'] = 'Times New Roman'
    return None

# Reading response and independent variablescase1/data/response.txt
y = np.loadtxt('case1/data/response.txt')
X = np.loadtxt('case1/data/independent.txt')

#Temp NaN solution, replace nan with column mean
X = np.where(np.isnan(X), np.ma.array(X, mask=np.isnan(X)).mean(axis=0), X) 

(n, p) = X.shape

CV = 5
kf = KFold(n_splits=CV, random_state=42, shuffle=True)
 
stop = n-math.ceil(n/CV)
K = range(stop)  # TODO: assert this part, why do we use this appraoch?

Err_tr = np.zeros((CV,len(K)))
Err_val = np.zeros((CV, len(K)))


for i, (train_index, test_index) in enumerate(kf.split(X)):
    # NOTE: If you normalize outside the CV loop the data implicitly carry information of the test data
    # We should perform CV "the right way" and keep test data unseen.

    # Center data, notice test is centered using training mean
    y_train, mu = centerData(y[train_index])
    y_test = y[test_index] - mu

    X_train, mu = centerData(X[train_index, :])
    X_test = X[test_index,:] - mu

    # Normalizing the data
    X_train, d = normalize(X_train)
    X_test = X_test / d

    # compute all LARS solutions inside each fold
    for p in K:
        model = Lars(n_nonzero_coefs=p, fit_path = False, fit_intercept = False)
        model.fit(X_train, y_train)

        # Predict with this model, and find error
        beta = model.coef_.ravel()
        y_hat_train = X_train @ beta
        y_hat_test = X_test @ beta
        
        Err_tr[i, p] = np.sqrt(((y_train - y_hat_train)**2).mean())
        Err_val[i, p] = np.sqrt(((y_test - y_hat_test)**2).mean())



err_tr = np.mean(Err_tr, axis=0) # mean training error over the CV folds
err_tst = np.mean(Err_val, axis=0) # mean test error over the CV folds
err_ste = np.std(Err_val, axis=0)/np.sqrt(CV) # Note: we divide with sqrt(n) to get the standard error as opposed to the standard deviation
p_OP = np.argmin(err_tst) #Best performing p determined by CV.

# 1-std rule
seMSE = np.std(err_tst, axis=0) / np.sqrt(K)
P = np.where(err_tst[p_OP] + seMSE[p_OP] > err_tst)[0] #TODO: validate inequality sign
#j = int(P[-1:])
j = int(P[0])
p_CV_1StdRule = [i for i in K][j]
print("CV lambda with 1-std-rule %0.2f" % p_CV_1StdRule)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=100)

ax1.plot(K, Err_tr.T, ':', alpha=0.5)
ax1.plot(err_tr, c='k', label='Average training error', linewidth=1.5)
ax1.set_title('Training error for each fold')
#ax1.set_yscale('log')
ax1.legend()
ax1.set_xlabel(r"non-zero $\beta$")
ax1.set_ylabel('RMSE')

ax2.plot(K, Err_val.T, ':', alpha=0.5)
ax2.plot(err_tst, c='k', label='Average training error', linewidth=1.5)
ax2.axvline(p_OP, label=r"$p^{*}$", linestyle='dashed', c='k', alpha=0.75)
ax2.axvline(p_CV_1StdRule, label=r"$p$ (CV 1Std rule)", linestyle='dashed', c='firebrick', alpha=0.75)
ax2.set_title('Test error for each fold')
#ax2.set_yscale('log')
ax2.legend()
ax2.set_xlabel(r"non-zero $\beta$")
ax2.set_ylabel('RMSE')
plt.show()