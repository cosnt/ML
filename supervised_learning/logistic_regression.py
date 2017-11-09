import numpy as np

from utils.structs import OptimizationParm, Data
from utils.operation import get_batch, sigmoid, multi2one, cross_entropy

'''
label is not one_hot
'''

class LogRegres(object):
    def __init__(self, data, parm):
        self.data = data
        self.parm = parm
        self.theta = None
        self.cost = []

    def fit(self):
        feat_num, classfy_num = self.data.train_set.shape[1], max(self.data.train_label)+1
        self.theta = np.zeros((feat_num, classfy_num))
        for step in range(self.parm.max_step):
            train_data_batch, train_label_batch = get_batch(self.data.train_data,
                                                            self.data.train_label, self.parm.batch_size)
            for index in range(classfy_num):
                one_label = multi2one(train_label_batch, index)
                self.theta[:,index] +=  self.parm.alpha*np.mean(
                    (one_label - sigmoid(np.matmul(train_data_batch,self.theta[:,index])))*train_data_batch,axis=0).T
            y_pred = sigmoid(np.matmul(train_data_batch,self.theta))
            loss = cross_entropy(y_pred,train_label_batch)
            self.cost.append(loss)

    def predict(self):
        y_pred = np.argmax(np.matmul(self.data.test_data,self.theta),axis=1)
        accuracy = np.mean(y_pred==self.data.test_label)
        print('the accuracy is :', accuracy)

def main():
    file_name = '/mnt/'
    data, parm = Data(file_name), OptimizationParm(500,0.3)
    model = LogRegres(data, parm)
    model.fit()
    model.predict()

if __name__ == '__mian__':
    main()    
