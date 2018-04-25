import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def read_text_file(filename):
    """
    读取文本数据
    :param filename: 输入文件地址
    :return: array 数组
    """
    df = pd.read_table(filename, header=None, encoding='gb2312', delim_whitespace=True)
    return df.values


def get_euclidean_metric(X):
    """
    获的欧式距离矩阵
    :param X: 不包含label的数据
    :return: 输出m*m的矩阵，m为样本个数。矩阵值为各个点之间的距离
    """
    sample_nums, feature_nums = X.shape
    result_matrix = np.zeros(shape=[sample_nums, sample_nums])
    matrix = lambda x:(np.transpose([x])-x)**2
    for index in range(feature_nums):
        result_matrix += matrix(X[:,index])
    return result_matrix

def distance_matrix(X, with_label = False,mode='euclidean'):
    """
    求解样本间距矩阵
    :param X: 输入数据，数组
    :param with_label: 数据是否包含标签，bool值
    :param mode: 度量函数，目前只写了欧式距离函数，可自行添加
    :return:距离矩阵、labels（如愿数据不包含label，则label值默认为0）
    """
    distance = None
    labels = np.zeros([X.shape[0]])
    if with_label:
        labels = X[:,-1]
        X = X[:,:-1]
    if mode == 'euclidean':
        distance =  get_euclidean_metric(X)
    else:
        pass
    return distance, labels

def get_dc(distance):
    """
    自动选择dc值
    :param distance: 距离举证
    :return: dc值
    """
    temp  = []
    for index, value in enumerate(distance):
        temp.extend(value[index+1:])
    sort_temp = np.sort(temp)
    index_k = round(0.02*len(sort_temp))
    return sort_temp[index_k]

def get_local_density_sigma(distance, dc, kernel = 'gaussian'):
    """
    求解局部密度
    :param distance: 距离矩阵
    :param dc: dc
    :param kernel: 求解模式
    :return: 各个点局部密度
    """
    density, density_matrix = None, None
    if kernel == 'gaussian':
        density_matrix = np.exp(-(distance/dc)**2)
        density = np.sum(density_matrix,axis=1)-1
    elif kernel == 'cut_off':
        density_matrix = np.ones(distance.shape)[distance<dc]
        density = np.sum(density_matrix, axis=1)-1
    return density

def get_sigma(distance, density):
    """
    求解sigma值
    :param distance: 距离矩阵
    :param density: 密度值
    :return: delta 各点离与最高密度点的距离
    """
    sample_nums = len(density)
    sigma = []
    for index in range(sample_nums):
        max_density = max(density)
        if max_density == density[index]:
            sigma.append(np.max(distance[index,:]))
        else:
            label = density > density[index]
            sigma.append(np.min(distance[index,label]))
    return sigma

def auto_select_centre(distance,density,sigma):
    """
    自动选取聚类中心。原文根据设置密度，最短距离的阈值来选择。暂未编写
    :param distance:距离矩阵
    :param density:密度
    :param sigma:最短距离
    :return: 聚类中心值
    """
    return 3

def auto_classfy(distance, density, sigma, peaks):
    """
    根据自动求解的聚类中心，进行聚类。可自动剔除异常点
    :param distance:距离矩阵
    :param density:密度
    :param sigma:最小距离
    :param peaks:聚类中心数目
    :return:聚类结果（数据对应类别）
    """
    value = sigma*density
    sort_index = np.argsort(-value)
    init_label = np.zeros(value.shape,dtype=np.int)
    index_density = np.argsort(-density)
    for index,value in enumerate(index_density[1:]):
        density_greater_distance = distance[value,:][index_density[:index+1]]
        index_density_greater_distance = np.argmin(density_greater_distance)
        init_label[value] = index_density[index_density_greater_distance]
    class_label = np.zeros(density.shape,dtype=np.int)-1
    for index,value in enumerate(sort_index[:peaks]):
        class_label[value] = index
    for value in index_density:
        if class_label[value] == -1:
            class_label[value] = class_label[init_label[value]]
    return class_label

def density_peak_cluster(X):
    """
    密度聚类函数，自动对数据进行聚类。无需指定聚类中心数目，能够自动处理异常点。
    :param X: 样本数据
    :return: 局部密度、最短距离、原始标签、聚类算法处理标签
    """
    distance, labels = distance_matrix(X,True)
    dc = get_dc(distance)
    local_density = get_local_density_sigma(distance,dc)
    sigma = get_sigma(distance,density=local_density)
    peaks = auto_select_centre(distance,local_density,sigma)
    class_label = auto_classfy(distance, local_density, sigma, peaks)
    return local_density, sigma, labels, class_label

def plot_scatter(X, density, sigma,labels,class_label):
    plt.figure(1)
    plt.scatter(X[:,0], X[:,1], c=labels)
    plt.figure(2)
    plt.scatter(density,sigma)
    plt.figure(3)
    value = density*sigma
    value_sorted = -np.sort(-value)
    plt.scatter(range(len(value_sorted)),value_sorted)
    plt.figure(4)
    plt.scatter(X[:,0],X[:,1],c=class_label)
    plt.show()

if __name__ == "__main__":
    filename = '../data_warehouse/local_data/spiral.txt'
    X = read_text_file(filename)
    local_density, sigma, labels, class_label = density_peak_cluster(X)
    plot_scatter(X, local_density, sigma,labels,class_label)
