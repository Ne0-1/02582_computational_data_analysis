import numpy as np
from functions import centerData, normalize
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer


X_new = np.loadtxt('../case1/exploratory/feature_matrix_new.txt')
cat_idx = [95, 97, 98, 99]
K = 5

# Imputing missing values
# Real valued features
imputer_par = KNNImputer(n_neighbors=K, weights='distance', metric='nan_euclidean')
imputer_par.fit(X_new[:, :95])
X_new[:, :95] = imputer_par.transform(X_new[:, :95])
for cvar in [95, 96, 97, 98, 99]:
    knn_par = KNeighborsClassifier(n_neighbors=K, weights='distance')
    #knn_test = KNeighborsClassifier(n_neighbors=K, weights='distance')

    par_miss = np.where(np.isnan(X_new[:, cvar]))[0]
    mask_par = np.ones(X_new.shape[0], dtype=bool)
    mask_par[par_miss] = False

    if par_miss.size != 0:
        knn_par.fit(X_new[mask_par, :95], X_new[mask_par, cvar])
        par_impute = knn_par.predict(X_new[par_miss, :95])
        X_new[par_miss, cvar] = par_impute


assert np.isnan(X_new).sum() == 0, "Something whent wrong with imputing par"
# Onehot encode categorical variables (Not binary variables)
enc = OneHotEncoder(sparse=False)
enc.fit(X_new[:, cat_idx])
X_new_cat_encoded = enc.transform(X_new[:, cat_idx])
# Note that order of non continoues variables have been reordered from [C1, C2, .., C5] -> [C2, C1, C3, C4, C5]

# Drop cat vars and stack continous and dummy coded
X_new = np.delete(X_new, cat_idx, axis=1)
X_new = np.hstack([X_new, X_new_cat_encoded])

# Center data, notice test is centered using training mean
X_new, mu = centerData(X_new)

# Normalize val using train
X_new, d = normalize(X_new)

np.savetxt('X_new_processed.txt', X_new)