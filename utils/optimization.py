# _*_ coding:utf-8 _*_
import numpy as np
'''
input:
    data
        trainSet, trianLabel
    parm:
        max_step
        leraning rate
        optimization mode
    theta:

output:
    theta, costs
'''

def sigmoid(X):
    return 1/(1+np.exp(-X))

def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))


def BGD(train_data, opt_parm, theta = None):
    train_num, feat_num = train_data[0].shape
    if theta is None:
        theta = np.zeros((feat_num, 1))
    step, cost = 0, []
    for index in range(opt_parm.max_step):
        theta = theta + ((1.0/train_num)*opt_parm.alpha*(train_data[1] - sigmoid(train_data[0]*theta)).T*train_data[0]).T \
                + (opt_parm.lam/train_num)*np.row_stack((0,theta[1:]))
        temp = sigmoid(train_data*theta)
        cost.append( ((-1.0/train_num)*(train_data[0].T*np.log(temp) + (1 - train_data[1]).T*np.log(1 - temp)) + (self.lam/(2*train_um))*theta.T*theta)[0,0])
        if (abs(cost[-1] - cost[-2]) < opt_parm.tolerance ):
            break
    return theta, cost
#stochastic gradient descent

def SGD(self, theta, trainData, trainLabel):
    index = range(self.trainNum)
    np.random.shuffle(index)
    cost = []
    for i in index:
        theta = theta + self.alpha*(self.sigmoid(trainLabel[i] - trainData[i,:]*theta.T))*trainData[i,:] \
            + self.lam*np.row_stack((0,theta[1:]))
        temp = self.sigmoid(trainData * theta.T)
        cost.append(trainLabel.T * np.log(temp) + (1 - trainLabel).T * np.log(1 - temp) + (self.lam/(2*self.trainNum))*theta.T*theta)
        if (iter > self.maxIter or cost[-1] < self.tolerance):
            break
    return theta

#mini batch gradient descent
def MBGD(self, theta, trainData, trainLabel):
    minSetNum = 20
    head = 0
    tail = 20
    cost = []
    while(True):
        theta = theta + (1/(tail - head))*self.alpha*(trainLabel[head:tail] - trainData[head:tail,:]*theta).T*trainData[head:tail,:] +\
                (self.lam/(tail - head))*np.row_stack((0,theta[1:]))
        temp = self.sigmoid(trainData * theta.T)
        cost.append(trainLabel.T * np.log(temp) + (1 - trainLabel).T * np.log(1 - temp) + ( self.lam / self.trainNum) * theta.T * theta)
        if (iter > self.maxIter or cost[-1] < self.tolerance):
            break
        head = tail
        tail = head + minSetNum
        if tail > self.trainNum:
            tail = self.trainNum
    return theta
