import pandas as pd
import numpy as np
import matplotlib as mpt
#import seaborn

def min_max(data):
    min_value = np.min(data, axis = 0)
    max_value = np.max(data, axis = 0)
    data = (data - min_value)/max_value
    return data, min_value, max_value

def z_score(data):
    mean_value = np.mean(data, axis = 0)
    std_value = np.std(data, axis = 0)
    data = (data - mean_value)/std_value
    return data, mean_value, std_value

def normalization(data, methon):
    if methon == "min_max":
        return min_max(data)
    elif methon == "z_score":
        return z_score(data)
    else:
        raise("the methon not support")

def norm_fit(data, parm):
    if parm.methon == "min_max":
        return (data - parm.min_value)/parm.max_value
    elif parm.methon == "z_score":
        return (data - parm.mean_value)/parm.std_value
    else:
        raise("the methon not support")

def transLabel(labels):
    label_set = set(labels)
    trans_label = dict()
    for key in label_set:
        trans_label[key] = trans(key, labels)
    return trans_label

def trans(key, labels):
    index = labels == key
    trans_labes = np.zeros(labels.shape)
    trans_labes[index] = 1
    return trans_labes

def split_data(data, train_rate = 0.7, test_rate = 0.3, cv_rate = 0):
    if train_rate == 0 or test_rate == 0:
        raise("the value of train_rate or test_rate can not to set 0")
    n_data = len(data)
    np.random.shuffle(data)
    n_train = int(np.round(n_data*train_rate))
    n_test = int(np.round(n_data*test_rate))
    train_set = data[:n_data,:]
    test_set = data[n_data:n_data+n_test,:]
    if cv_rate != 0:
        n_cv = int(np.round(n_data * cv_rate))
        cv_set = data[n_data+n_test:,:]
        return train_set, test_set, cv_set
    return train_set, test_set

def data_cleaning():
    #补全
    pass
