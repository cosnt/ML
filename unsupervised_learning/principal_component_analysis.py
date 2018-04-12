import numpy as np
from utils.operation import normalization

def pca(data, k=2, tolerance = None):
    data = normalization(data,'Standardization')
    sigma = data.T*data
    U, lambdas, V = np.linalg.svd(sigma)
    if tolerance is not None:
        lambda_sum = sum(lambdas)
        for index in range(len(lambdas)):
            if sum(lambdas[:index+1])/lambda_sum:
                U = U[:,:index+1]
    else:
        U = U[:,:k]
    return data.T*U, U

def kernel_pca(data, k=2, tolerance=None):
    pass