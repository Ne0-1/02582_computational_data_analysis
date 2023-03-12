import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
from functions import normalize, centerData
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
import warnings

def setup_mpl():
    mpl.rcParams['font.family'] = 'Times New Roman'
    return None

# Reading response and independent variablescase1/data/response.txt
y = np.loadtxt('case1/data/response.txt')
X = np.loadtxt('case1/data/independent.txt')

#Temp NaN solution, replace nan with column mean
X = np.where(np.isnan(X), np.ma.array(X, mask=np.isnan(X)).mean(axis=0), X) 

(n, p) = X.shape

#Sklearn have switch the meaning of alpha and lambda (https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
#The parameter l1_ratio corresponds to alpha in the glmnet R package while alpha corresponds to the lambda parameter in glmnet.

# GLMNET R interpretation #
# ----------------------- # 
# General notes to understand hyperparameters
# penalty = (alpga * l1) + (1-alpha)*l2
# elastic_net_loss = loss + (lambda * penalty)

# SKLEARN interpretation #
# ---------------------- #
#1 / (2 * n_samples) * ||y - Xw||^2_2 + alpha * l1_ratio * ||w||_1 + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2 (Expanded)
# 1 / (2 * n_samples) * ||y - Xw||^2_2 + (alpha * penalty)

alphas = np.linspace(0, 1, 5)
lambdas = np.logspace(-3, 1, 20)

for alpha in alphas:
    # Perform K-fold cross validation
    CV = 5
    kf = KFold(n_splits=CV, random_state=42, shuffle=True)

    coefs = np.zeros((CV,len(lambdas),p))
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        
        # Center data, notice test is centered using training mean
        y_train, mu = centerData(y[train_index])
        y_test = y[test_index] - mu

        X_train, mu = centerData(X[train_index, :])
        X_test = X[test_index,:] - mu

        # Normalizing the data
        X_train, d = normalize(X_train)
        X_test = X_test / d
        
        for j, lambda_ in enumerate(lambdas):
            with warnings.catch_warnings(): # disable convergence warnings from elastic net
                warnings.simplefilter("ignore")
                model = ElasticNet(alpha=lambda_, l1_ratio=alpha)
                model.fit(X_train, y_train)
                coefs[i,j,:] = model.coef_
            
    trace = np.sum(coefs, axis=0)

    plt.figure()
    plt.semilogx(lambdas, trace)
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Sum of coefficients')
    plt.title('Sum of coefficients of Elastic Net Fit Alpha = %.3f' % alpha)
    plt.show()



