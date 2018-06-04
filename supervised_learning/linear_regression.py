import re
import os
import sys
import numpy as np

from utils.operation import mean_squared_error, get_batch
from utils.structs import OptimizationParm, Data

class OptimizationParameters(object):
    def __init__(self, max_step, learning_rate):
        self.alpha = learning_rate
        self.max_step = max_step

class LinerRegre(object):
    def __init__(self, data, parm):
        self.cost = []
        self.data = data
        self.parm = parm
        self.theta = np.zeros((self.data.train_data.shape[1], 1))

    def fit(self):
        for step in range(self.parm.max_step):
            batch_train_data, batch_trian_label = get_batch(self.data.trian_data, self.data.train_label, self.parm.batch)
            predict_value = np.matmul(self.data.trainSet, self.theta)
            self.theta += self.parm.alpha*np.mean((predict_value - batch_trian_label)*batch_train_data,axis=0)
            loss = mean_squared_error(predict_value,batch_trian_label)
            self.cost.append(loss)

    def predict(self):
        predict_value = np.matmul(self.data.testSet, self.theta)
        loss = mean_squared_error(predict_value, self.data.testLabel)
        print('the loss is: ', loss)

def read_txt(filename):
    with open(filename, 'r') as fr:
        lines = fr.readlines()
        data = np.zeros((len(lines),len(lines[0].split())))
        for index, line in enumerate(lines):
            data[index,:] = list(map(float,re.split(' |,',line)))
    return data


dataSet = read_txt('./dataSet/flame.txt')
dataSet[:,-1] += 1

def main():
    file_name = ''
    data = Data(file_name)
    parm = OptimizationParm(500, 0.3)
    model = LinerRegre(data, parm)
    model.fit()
    model.predict()

if __name__ == '__main__':
    main()
