import numpy as np
from scipy.linalg import eigh

class PCA:
    def __init__(self):
        self.mean=None
        self.eigen_vector=None
        self.output=None

    def fit(self, X, dim=3):
        n, num_X = X.shape[0], X.shape[1] # N=400, n=10304
        dim = 3
        mean_X = np.mean(X, axis=1).reshape(-1, 1) / num_X # mx: n * 1
        X = X - mean_X
        cx = np.dot(X, X.T) # cx : n * n
        _, eigen_vector = eigh(cx, subset_by_index=[n-dim, n-1])
        output = np.dot(X.T, eigen_vector)
        
        self.mean=mean_X
        self.eigen_vector=eigen_vector
        self.output=output
        return mean_X, eigen_vector, output
