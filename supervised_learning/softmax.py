import numpy as np
from utils.operation import get_batch, sigmoid, multi2one, code_one_hot
from utils.structs import OptimizationParm, Data
'''softmax
'''
class MutilLayerPreceptron(object):
    def __init__(self, data, parm, shape):
        self.data = data
        self.parm = parm
        self.theta = None
        self.net_shape = shape
        self.cost = []

    def fit(self):
        feat_num, classfy_num = self.data.train_set.shape[1], max(self.data.train_label)
        self.theta = np.zeros((feat_num, classfy_num))
        for step in range(self.parm.max_step):
            train_data_batch, train_label_batch = get_batch(self.data.train_data,
                                                            self.data.train_label, self.parm.batch_size)
            for index in range(classfy_num):
                one_label = multi2one(train_label_batch, index)
                temp = np.exp(np.matmul(train_data_batch,self.theta[:,index]))/\
                       np.sum(np.matmul(train_data_batch,self.theta),axis=1)
                self.theta[:, index] += self.parm.alpha*np.mean((one_label - temp)*train_data_batch, axis=0).T
            one_hot = code_one_hot(train_label_batch, classfy_num+1)
            temp = np.matmul(train_data_batch,self.theta)
            loss = np.mean(np.sum(one_hot*( temp - np.log(np.sum(np.exp(temp),axis=1))),axis=1),axis=0)
            self.cost.append(loss)

    def predict(self):
        temp = np.matmul(self.data.test_data, self.theta)
        y_pred = np.argmax(temp/np.sum(temp,axis=1), axis=1)
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

