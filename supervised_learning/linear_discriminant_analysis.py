import numpy as np

from utils.structs import Data, OptimizationParm

class OptimizationParameters(object):
    def __init__(self, max_step, learning_rate):
        self.max_step = max_step
        self.alpha = learning_rate

class LDA(object):
    def __init__(self, data, parm):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

def main():
    file_name = '/mnt/'
    data, parm = Data(file_name), OptimizationParm(500,0.3)
    model = LDA(data, parm)
    model.fit()
    model.predict()

if __name__ == '__mian__':
    main()