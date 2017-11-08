import sys
import math
import numpy as np


'''
basic function
array*array point
matmul(a,b) mat
'''
def sigmoid(X):
    return 1/(1+np.exp(-X))

def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))

def accuracy_score(y_true, y_pred):
    correct = 0
    for i in range(len(y_true)):
        diff = y_true[i] - y_pred[i]
        if diff == np.zeros(np.shape(diff)):
            correct += 1
    return correct / len(y_true)

def normalization(data):
    if data.mode == 'max_':
        if data.max_ is None:
            data.max_ = np.max(data.trainSet,0)
        data.trianSet = data.trianSet/data.max_
    elif data.mode == 'max_min':
        max_ = np.max(data.trainSet,axis=0)
        min_ = np.min(data.trainSet, axis = 0)
        data.trainSet = (data.trainSet - min_)/(max_ - min_)
    elif data.mode == 'Standardization':
        data.mean = np.mean(data.trianSet, axis = 0)
        data.std = np.std(data.trianSet, axis = 0)
        data.trainSet = (data.trainSet -data.mean)/data.std

def data_fit(data):
    numberOfColumns = len(data.testSet)
    if data.mode == 'max_':
        if data.max_ is None:
            data.max_ = np.max(data.testSet,0)
        data.testSet = data.testSet/data.max_
    elif data.mode == 'max_min':
        max_ = np.max(data.testSet,axis=0)
        min_ = np.min(data.testSet, axis = 0)
        data.testSet = (data.testSet - min_)/(max_ - min_)
    elif data.mode == 'Standardization':
        data.testSet = (data.testSet - data.mean)/data.std

def get_batch(train_data, train_label, batch_size):
    row_num = train_data.shape[0]
    random_index = np.random.choice(row_num, batch_size)
    data_batch = train_data[random_index, :]
    label_batch = train_label[random_index]
    return data_batch, label_batch

def calculate_entropy():
    pass

def mean_squared_error(y_true, y_pred):
    if y_true.shape[1] == 1:
        y_true = y_true.T
    if y_pred.shape[1] == 1:
        y_pred = y_pred.T
    temp = y_true - y_pred
    return 0.5*temp*temp.T

def euclidean_distance(x1, x2):
    temp = x1 - x2
    return np.sqrt(temp*temp.T)

def code_one_hit(label, classfy_num):
    set_lens = len(label)
    one_hot = np.zeros((set_lens, classfy_num))
    for index in range(label):
        one_hot[index, label[index]] = 1
    return one_hot

def multi2one(labels,classfy_no):
    one_label = np.zeros(labels.shape)
    one_label[labels==classfy_no] = 1
    return one_label

def cross_entropy(y_pred, y_true):
    temp = np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred),axis=0)
    return np.sum(temp,axis=1)

def split_data(data):
    pass

def loss():
    pass

def k_folder(data, label, K):
    pass

def ROC(prediction_label, actual_label):
    pass

def loss_curve(prediction_label, actual_label):
    pass

def accuracy_curve(prediction_label, actual_label):
    pass

def accuracy_rate(prediction_label, actual_label):
    pass

def precision_rate(prediction_label, actual_label):
    pass

def recall_rate(prediction_label, actual_label):
    pass

def specificity_rate(prediction_label, actual_label):
    pass

def F1_socore(prediction_label, actual_label):
    pass

def MCC(prediction_label, actual_label):
    pass

def AUC(prediction_label, actual_label):
    pass
