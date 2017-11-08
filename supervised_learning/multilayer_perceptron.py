import numpy as np
from utils.operation import mean_squared_error, get_batch
from utils.structs import OptimizationParm, Data
'''
softmax
'''
class MutilLayerPreceptron(object):
    def __init__(self, data, parm, shape):
        self.data = data
        self.parm = parm
        self.connect_parm = self._init_connect_parm(shape)
        self.cost = []

    def _init_connect_parm(self, shape):
        connect_parm = {}
        for index in range(len(shape)-1):
            connect_parm['w'+str(index)] = np.zeros((shape[index],shape[index+1]))
            connect_parm['b'+str(index)] = np.zeros((shape[index],1))
        return connect_parm

    def fit(self):
        pass

    def predict(self):
        pass

def main():
    file_name = ''
    data = Data(file_name)
    parm = OptimizationParm(500, 0.3)
    shape = [100,1024,10]
    model = MutilLayerPreceptron(data, parm, shape)
    model.fit()
    model.predict()
if __name__ == '__main__':
    main()

