import numpy as np

from utils.structs import Data, OptimizationParm

class OptimizationParameters(object):
    pass

class KNN(object):
    def __init__(self, data, parm):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

def main():
    file_name = '/mnt/'
    data, parm = Data(file_name), OptimizationParm(500,0.3)
    model = KNN(data, parm)
    model.fit()
    model.predict()

if __name__ == '__mian__':
    main()