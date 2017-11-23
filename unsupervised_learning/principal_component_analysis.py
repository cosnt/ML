import numpy as np
from utils.operation import normalization

def pca(data, k=2, tolerance = 0.9):
    data = normalization(data,'Standardization')
    m = len(data)
    sigma = (1/m)*(data.T*data)
    U, Sigma, V = np.linalg.svd(sigma)
