import numpy as np

from utils.structs import Data, OptimizationParm

class OptimizationParameters(object):
    pass

'''
伯努利
多项式
高斯分布
基本版2分类
'''

class NaiveBayes(object):
    def __init__(self, data, parm):
        self.data = data
        self.parm = parm

    def fit(self, mode='bernoulli'):
        if mode == 'bernoulli':
            self._fit_bernoulli()
        elif mode == 'polynomial':
            self._fit_polynomial()
        elif mode == 'gaussian':
            self._fit_gaussian()
        else:
            print('not support')

    def _fit_bernoulli(self):
        self.alpha = np.mean(self.data.train_label)
        self.theta_y0 = np.sum(self.data.trian_data*self.data.train_label,axis=0)/np.sum(self.data.train_label)
        self.theta_y1 = np.sum((1-self.data.train_data)*self.data.train_label,axis=0)/np.sum(self.data.train_label)

    def _fit_polynomial(self):
        pass

    def _fit_gaussian(self):
        pass

    def predict(self, mode):
        if mode == 'bernoulli':
            self._predict_bernoulli()
        elif mode == 'polynomial':
            self._predict_polynomial()
        elif mode == 'gaussian':
            self._predict_gaussian()
        else:
            print('not support')

    def _predict_bernoulli(self):
        predict_y0 = (1-self.alpha)*np.product(self.data.test_data*self.theta_y0,axis =1)
        predict_y1 = self.alpha*np.product(self.data.test_data*self.theta_y1,axis = 1)
        predict = predict_y1/(predict_y0+predict_y0)
        predict_label = np.zeros((self.data.test_label))
        predict_label[predict>0.5] = 1
        accuracy = np.mean(predict_label==self.data.test_label)
        print("the accuracy is :",accuracy)

    def _predict_polynomial(self):
        pass

    def _predict_gaussian(self):
        pass

def main():
    file_name = '/mnt/'
    data, parm = Data(file_name), OptimizationParm(500,0.3)
    model = NaiveBayes(data, parm)
    model.fit()
    model.predict()

if __name__ == '__mian__':
    main()