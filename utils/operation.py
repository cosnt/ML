import sys
import math
import numpy as np

'''
basic function
array*array point
matmul(a,b) mat
'''

#激活函数
###################################################
def sigmoid(X):
    return 1/(1+np.exp(-X))

def sigmoid_gradient(x):
    return sigmoid(x) * (1 - sigmoid(x))

####################################################
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

def code_one_hot(label, classes=None):
    if classes is None:
        classes = np.unique(label)
    set_lens = len(label)
    one_hot = np.zeros((set_lens, len(classes)))
    for index in range(set_lens):
        one_hot[index, label[index]] = 1
    return one_hot

def multi2one(labels,classfy_no):
    one_label = np.zeros(labels.shape)
    one_label[labels==classfy_no] = 1
    return one_label

def cross_entropy(y_pred, y_true):
    temp = np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred),axis=0)
    return np.sum(temp,axis=1)

##################distancs function################################
#matrix
def similarity(test_data, train_data, mode = "euclidean"):
    test_data_num, train_data_num = len(test_data), len(train_data)
    similarity_matrix = np.zeros((test_data_num, train_data_num))
    if mode == "euclidean":
        for index in range(test_data_num):
            similarity_matrix[index,:] = euclidean_distance(test_data[index,:],train_data)
    elif mode == 'cosine':
        for index in range(test_data_num):
            similarity_matrix[index,:] = cosine_distance(test_data[index,:],train_data)
    elif mode == 'pearson':
        for index in range(test_data_num):
            similarity_matrix[index,:] = pearson_distance(test_data[index,:],train_data)
    return similarity_matrix

def euclidean_distance(test_data, train_data):
    kernel = train_data - test_data
    return np.sqrt(np.sum(np.power(kernel,2),axis =1)).T

def cosine_distance(test_data,train_data):
    temp_a = np.sqrt(np.sum(np.power(test_data,2),axis=1))
    temp_b = np.sqrt(np.sum(np.power(train_data,2),axis=1))
    return train_data*test_data.T/(temp_a*temp_b)

def pearson_distance(test_data,trian_data):

    pass


####################################################################

def split_data(data, rate = 0.7):
    sample_nums = len(data.data)
    np.random.shuffle(data.data)
    train_index = int(rate*sample_nums)
    data.train_data = data.data[:train_index,:-1]
    data.train_label = data.data[:train_index:,-1]
    data.test_data = data.data[train_index:,:-1]
    data.test_label = data.data[train_index:,-1]

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
