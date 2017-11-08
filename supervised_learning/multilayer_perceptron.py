import numpy as np
from utils.operation import mean_squared_error, get_batch, sigmoid
from utils.structs import OptimizationParm, Data
'''
softmax
'''
class MutilLayerPreceptron(object):
    def __init__(self, data, parm, shape):
        self.data = data
        self.parm = parm
        self.connect_parm = self._init_connect_parm(shape)
        self.net_shape = shape
        self.cost = []

    def _init_connect_parm(self, shape):
        connect_parm = {}
        for index in range(len(shape)-1):
            connect_parm['w'+str(index)] = np.zeros((shape[index],shape[index+1]))
            connect_parm['b'+str(index)] = np.zeros((shape[index],1))
        return connect_parm

    def fit(self):
        classfy_num = max(self.data.train_label)
        for step in range(self.parm.max_step):
            train_data_batch, train_label_batch = get_batch(self.data.train_data,
                                                            self.data.train_label, self.parm.batch_size)
        for index in range(classfy_num):
            pass


    def predict(self):
        out_value = self.data.testSet
        for index in range(len(self.net_shape)-1):
            out_value = sigmoid(np.matmul(out_value,self.connect_parm['w'+str(index)]) +
                                self.connect_parm['b'+str(index)])
        y_pred = np.argmax(out_value, axis=1)
        accuracy = np.mean(y_pred == self.data.testLabel)
        print('the accuracy is :', accuracy)

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

