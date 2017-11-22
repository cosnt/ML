import numpy as np

from utils.structs import Data, OptimizationParm
from utils.operation import similarity


class OptimizationParameters(object):
    pass

class KNN(object):
    def __init__(self, data, K):
        self.data = data
        self.K = K

    def fit(self):
        pass

    def predict(self):
        pass

    def predict_simple(self):
        distance = similarity(self.data.test_data, self.data.train_data)
        distance_index = np.argsort(-distance, axis=1)[:,:self.K]
        labels = self.data.train_data[distance_index]


def main():
    file_name = '/mnt/'
    data, parm = Data(file_name), OptimizationParm(500,0.3)
    model = KNN(data, parm)
    model.fit()
    model.predict()

if __name__ == '__mian__':
    main()