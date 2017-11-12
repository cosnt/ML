import numpy as np

from utils.structs import Data, OptimizationParm
from utils.operation import code_one_hot


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
        self.conditional_probability = {}

    def fit(self, mode='bernoulli'):
        if mode == 'bernoulli':
            self._fit_bernoulli()
        elif mode == 'polynomial':
            self._fit_polynomial()
        elif mode == 'gaussian':
            self._fit_gaussian()
        elif mode == 'gda':
            self._fit_gaussian_discriminant_analysis()
        else:
            print('not support')

    def _fit_bernoulli(self):
        self.alpha = np.mean(self.data.train_label)
        self.theta_y0 = np.sum(self.data.trian_data*self.data.train_label,axis=0)/np.sum(self.data.train_label)
        self.theta_y1 = np.sum((1-self.data.train_data)*self.data.train_label,axis=0)/np.sum(self.data.train_label)

    def _fit_polynomial(self):
        #multi-multi
        label_one_hot = code_one_hot(self.data.train_label)
        self.alpha = np.mean(label_one_hot,axis=0)
        feat_num = len(self.data.train_data[0])
        #取消循环，三层矩阵（否定，矩阵不一定一样）
        for index in range(feat_num):
            #单个特征与类的条件概率
            feat_one_hot = code_one_hot(self.data.train_data[:,index])
            self.conditional_probability[str(index)] = np.matmul(feat_one_hot.T,label_one_hot)/\
                                                       np.sum(label_one_hot,axis=0)
    def _fit_gaussian(self):
        #以类别产生矩阵
        label_one_hot = code_one_hot(self.data.test_lebel)
        feat_num = len(self.data.train_data[0])
        self.alpha = np.mean(label_one_hot, axis=0)
        classes_num = len(np.unique(self.data.train_label))
        self.gaussian_mean = np.zeros((classes_num,feat_num))
        self.gaussian_std = np.zeros((classes_num,feat_num))
        for index in range(classes_num):
            temp = self.data.test_data*label_one_hot[:,index]
            self.gaussian_mean[index,:] = np.mean(temp, axis=0)
            self.gaussian_std[index, :] = np.std(temp, axis=0)

    def _fit_gaussian_discriminant_analysis(self):
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
        classes = np.unique(self.data.test_label)
        test_nums, classes_num = self.data.test_data.shape[0], len(classes)
        feat_num = self.data.test_data.shape[1]
        predict_probability = np.ones((test_nums,classes_num))
        for index in range(feat_num):
            feat_one_hot = code_one_hot(self.data.test_data[:,index])
            predict_probability *= (feat_one_hot*self.conditional_probability[str(index)])
        predict_probability *= self.alpha
        predict_probability = predict_probability/np.sum(predict_probability,axis=1)
        predict_label = np.argmax(predict_probability,axis=1)
        accuracy = np.mean(predict_label == self.data.test_label)
        print('the accuracy is :',accuracy)

    def _predict_gaussian(self):
        classes = np.unique(self.data.test_label)
        test_nums, classes_num = self.data.test_data.shape[0], len(classes)
        feat_num = self.data.test_data.shape[1]
        predict_probability = np.zeros((test_nums, classes_num))
        for index in range(feat_num):
            predict_probability *= self._get_gaussian_probability(index)
        predict_probability *= self.alpha
        predict_probability = predict_probability / np.sum(predict_probability, axis=1)
        predict_label = np.argmax(predict_probability, axis=1)
        accuracy = np.mean(predict_label == self.data.test_label)
        print('the accuracy is :', accuracy)

    def _get_gaussian_probability(self, index):
        probability = (1/(np.sqrt(2*np.pi*self.gaussian_std[index,:])))\
                      *np.exp(-np.power((self.data.test_data-self.gaussian_mean[index,:]),2)/(2*self.gaussian_std[index,:]))
        return probability

def main():
    file_name = '/mnt/'
    data, parm = Data(file_name), OptimizationParm(500,0.3)
    model = NaiveBayes(data, parm)
    model.fit()
    model.predict('bernoulli')

if __name__ == '__mian__':
    main()