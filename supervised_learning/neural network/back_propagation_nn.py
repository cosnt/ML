#L-BFGS
import numpy as np
from numpy.matlib import *
EPSILON = 10**-4


class BP():
    def __init__(self, data, classSet, nodes, alpha = 1, maxIter = 500, tolerance = 10**-10):
        self.trainSet = data["trainSet"]
        self.trainLabel = data["trainLabel"]
        self.classSet = classSet
        self.classNum = len(classSet)
        self.layerNum = len(nodes) + 2
        self.alpha = alpha
        self.maxIter = maxIter
        self.tolerance  = tolerance
        self.nodes = nodes
        self.weights = {}
        self.init()

    def init(self):
        self.dataNum, self.featNum = self.trainSet.shape
        self.weightInit()

    def weightInit(self):
        epsilon_init = 0.12
        self.nodes.insert(0, int(self.featNum))
        self.nodes.append(self.classNum)
        for i in range(1, self.layerNum):
            self.weights["W" + str(i)] = (2*rand((self.nodes[i], self.nodes[i - 1])) - 1) * epsilon_init
            self.weights["b" + str(i)] = epsilon_init*(2*rand((self.nodes[i], 1)) - 1 )

    def sigmoid(self, x):
        return 1.0/(1 + np.exp(-x))

    def sigmoidGrad(self, x):
        temp = self.sigmoid(x)
        return np.multiply(temp, (1 - temp))

#totle  matrix
    def activation(self):
        activations= {"a1" :self.trainSet}
        for i in range(2, self.layerNum + 1):
            activations["z" + str(i)] = activations["a" + str(i - 1)] * self.weights["W" + str(i - 1)].T \
                                        + repmat(self.weights["b" + str(i-1)].T, self.dataNum ,1)
            activations["a" + str(i)] = self.sigmoid(activations["z" + str(i)])
        return activations

    def errorTerm(self, activations):
        temp = activations["a" + str(self.layerNum)]
        errorTerm = {"delt" + str(self.layerNum) : np.multiply((temp - self.trainLabel), self.sigmoidGrad(temp))}
        for i in range(self.layerNum - 1, 1, -1):
            temp = activations["z" + str(i)]
            errorTerm["delt" + str(i)] = np.multiply( errorTerm["delt" + str(i + 1)] * self.weights["W" + str(i)], self.sigmoidGrad(temp) )
        return errorTerm

    def fit(self):
        iter = 0
        cost = []
        while(True):
            active = self.activation()
            errorTerm = self.errorTerm(active)
            self.weightsUpdate(active, errorTerm)
            iter = iter + 1
            cost.append( self.costFun( active["a" + str(self.layerNum)] ))
            if (iter >= self.maxIter or abs(cost[-1] - cost[-2]) < self.tolerance):
                break
        return self.weights, cost

    def weightsUpdate(self, active, errorTerm):
        for i in range(1, self.layerNum):
            self.weights["W" + str(i)] = self.weights["W" + str(i)] - \
                                         self.alpha*(( (1.0/self.dataNum)*errorTerm["delt" + str(i + 1)].T
                                                      * active["a" + str(i)] + 0.3*self.weights["W" + str(i)]))
            self.weights["b" + str(i)] = self.weights["b" + str(i)] - (1.0/self.dataNum)*self.alpha*(np.sum(errorTerm["delt" + str(i + 1)],0).T)

    def costFun(self, preValue):
        cost = 0
        for i in range(1, self.layerNum):
            temp = self.weights["W" + str(i)]
            cost = cost + 0.5*np.sum( np.multiply(temp, temp) )
        temp = self.trainLabel - preValue
        cost = cost + (0.5/self.dataNum)*np.sum( np.multiply(temp, temp))
        return cost
