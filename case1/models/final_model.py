import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import matplotlib as mpl
import matplotlib.pyplot as plt
from functions import centerData, normalize
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer

def setup_mpl():
    mpl.rcParams['font.family'] = 'Times New Roman'
    return None

# Reading response and independent variablescase1/data/response.txt
y = np.loadtxt('../case1/data/response_variable_y.txt')
X = np.loadtxt('../case1/data/feature_matrix_X.txt')

cat_idx = [95, 97, 98, 99]
K = 5

# Imputing missing values
# Real valued features
imputer_par = KNNImputer(n_neighbors=K, weights='distance', metric='nan_euclidean')
imputer_par.fit(X[:, :95])
X[:, :95] = imputer_par.transform(X[:, :95])
for cvar in [95, 96, 97, 98, 99]:
    knn_par = KNeighborsClassifier(n_neighbors=K, weights='distance')
    #knn_test = KNeighborsClassifier(n_neighbors=K, weights='distance')

    par_miss = np.where(np.isnan(X[:, cvar]))[0]
    mask_par = np.ones(X.shape[0], dtype=bool)
    mask_par[par_miss] = False

    if par_miss.size != 0:
        knn_par.fit(X[mask_par, :95], X[mask_par, cvar])
        par_impute = knn_par.predict(X[par_miss, :95])
        X[par_miss, cvar] = par_impute


assert np.isnan(X).sum() == 0, "Something whent wrong with imputing par"
# Onehot encode categorical variables (Not binary variables)
enc = OneHotEncoder(sparse=False)
enc.fit(X[:, cat_idx])
X_cat_encoded = enc.transform(X[:, cat_idx])
# Note that order of non continoues variables have been reordered from [C1, C2, .., C5] -> [C2, C1, C3, C4, C5]

# Drop cat vars and stack continous and dummy coded
X = np.delete(X, cat_idx, axis=1)
X = np.hstack([X, X_cat_encoded])

# Center data, notice test is centered using training mean
X, mu = centerData(X)
y, y_mu = centerData(y)


# Normalize val using train
X, d = normalize(X)


# Train optimal model
model_outer = Lasso(alpha=0.25595479, fit_intercept = False, tol=1e-4)
model_outer.fit(X, y)

# Training predictions
#beta_outer = model_outer.coef_.ravel()
#y_hat = X @ beta_outer # Adding intercept as beta0 = E[y]
y_hat = y_mu +  model_outer.predict(X)


Err = np.sqrt(((y - y_hat)**2).mean())
residuals = y-y_hat

fig, ax = plt.subplots(dpi=100)
ax.scatter(y_hat, residuals, s=8, c='black')
ax.axhline(0, linestyle='dashed', c = 'lightcoral', alpha=0.95)
ax.set_xlabel('Fitted values')
ax.set_ylabel('Residuals')
#plt.savefig('residuals.png', dpi=100)
#plt.show()

# final predictions
X_new = np.loadtxt('../case1/data/X_new_processed.txt')
#final_predictions = X_new @ beta_outer # Adding intercept as beta0 = E[y]
final_predictions = y_mu + model_outer.predict(X_new)
np.savetxt('predictions_s194266s194244.txt', final_predictions)
print(final_predictions.max())
print(final_predictions.min())
print(final_predictions.mean())
print(y_mu)


#fig, (ax1, ax2) = plt.subplots(1,2)
#ax1.plot(y)
#ax2.plot(final_predictions)
#plt.show()


