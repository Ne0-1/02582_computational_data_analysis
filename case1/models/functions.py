import numpy as np
def normalize(X):
    '''
    Function for normalizing the columns (variables) of a data matrix to unit length.
    Returns the normalized data and the L2 norm of the variables

    Input  (X) --------> The data matrix to be normalized
    Output (X_pre)-----> The normalized data matrix
    Output (d) --------> Array with the L2 norms of the variables
    '''
    d = np.linalg.norm(X, axis=0, ord=2)  # d is the euclidian lenghts of the variables
    d[d == 0] = 1  # Avoid dividing by zero if column L2 norm is zero
    X_pre = X / d  # Normalize the data with the euclidian lengths
    return X_pre, d


def centerData(data):
    mu = np.mean(data, axis=0)
    data = data - mu

    return data, mu