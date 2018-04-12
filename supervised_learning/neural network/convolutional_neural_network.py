#每层定义卷积子个数
#定义卷积子，池化子大小
#暂时默认步长1
#图像定义

import numpy as np
from numpy.matlib import *

class CNN():
    def __init__(self, data, paras):
        self.initData(data)
        self.initParas(paras)

    def initData(self, data):
        self.trainSet = data["trainSet"]
        self.trainLabel = data["trainLabel"]
        self.testSet = data["testSet"]
        self.testLabel = data["testLabel"]

    def initParas(self, paras):
        self.maxIter = paras["maxIter"]
        #self.conv = paras["conv"]
        #self.pool = paras["pool"]
        self.classSet = paras["classSet"]
        self.classNum = len(self.classSet)
        self.nodes = paras["nodes"]
        self.layerNum = len(self.nodes)

    def initWeight(self, nodes):
        epsilon_init = 0.12
        for i in range(self.layerNum):
            self.conv["conv" + str(i+1)] = (2*rand((self.nodes[i + 1], self.nodes[i])) - 1) * epsilon_init
            self.pool["pool" + str(i+1)] = epsilon_init*(2*rand((self.nodes[i + 1], 1)) - 1 )

    def convFunc(self, data,conv):
        convSize = conv.shape
        dataSize = data.shape
        xLength = dataSize[0] - convSize[0] + 1
        yLength = dataSize[1] - convSize[1] + 1
        convImage = np.matlib.zeros((xLength, yLength))
        for i in range(xLength):
            for j in range(xLength):
                convImage[i, j] = np.sum(np.multiply(data[i:i + convSize[0], j:j+ convSize[1]], conv))
        return convImage

    def poolFunc(self, data, pool):
        poolSize = pool.shape
        dataSize = data.shape
        poolImage = np.matlib.zeros((dataSize[0]/poolSize[0], dataSize[1]/poolSize[1]))
        for i in range(0, dataSize[0], poolSize[0]):
            for j in range(0, dataSize[1], poolSize[1]):
                poolImage[i/2, j/2] = np.sum(np.multiply(data[i: i+poolSize[0], j:j+poolSize[1]], pool))
        return poolImage

    def initWeight(self, shape):
        weight = 0.1*np.matlib.randn((shape[0], shape[1]))
        return weight

    def initBias(self, shape):
        bias = 0.1*np.matlib.ones((shape[0], shape[1]))
        return bias

    def convLayer(self, data):
        pass

    def poolLayer(self, data):
        pass

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoidGrad(self,x):
        temp = self.sigmoid(x)
        return np.multiply(temp, (1 - temp))

    def relu(self, x):
        pass

    def fit(self):
        pass
