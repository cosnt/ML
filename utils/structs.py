import os
import numpy as np

class OptimizationParm(object):
    def __init__(self, max_step, leraning_rate, lambda_ = 0, mode = 'gradient_descent'):
        self.max_step = max_step
        self.alpha = leraning_rate
        self.lambda_ = lambda_
        self.mode = mode

class Data(object):
    def __init__(self, file_name, data_shape = None, data_type = None):
        self.data = self.load_data(file_name, data_shape, data_type)
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None
        self.norm_mode = None
        self.mena = None
        self.std = None
        self.max_ = None
        self.split_mode = None

    def load_data(self, file_name, data_shape, data_type):
        file_format = file_name.split('.')[-1]
        temp_name = ''.join(file_name.split('.')[:-1]) + '.bin'
        if os.path.isfile('../data_warehouse/local_data/'+ temp_name):
            self.load_bin(file_name, data_shape, data_type)
        elif file_format == 'csv' or 'txt':
            self.load_csv(file_name)
        elif file_format == 'txt':
            self.load_txt(file_name)

    def load_csv(self, file_name):
        with open(file_name, 'rb') as reader:
            lines = reader.readlines()
            set_lens, feat_num = len(lines), len(lines[0])
            self.data = np.zeros((set_lens, feat_num))
            for index in range(set_lens):
                temp = lines[index]
                self.data[index,:] = temp
        self.save_as_bin(file_name)

    def load_txt(self,file_name):
        with open(file_name, 'rb') as reader:
            lines = reader.readlines()
            set_lens, feat_num = len(lines), len(lines[0])
            self.data = np.zeros((set_lens, feat_num))
            for index in range(set_lens):
                temp = lines[index]
                self.data[index,:] = temp
        self.save_as_bin(file_name)

    def load_bin(self, file_name, data_shape, data_type):
        file_name = '../data_warehouse/local_data/' + file_name
        self.data = np.fromfile(file_name, dtype=data_type)
        self.data.shape = data_shape

    def save_as_bin(self, file_name, fileparm_name = 'file_parm.txt'):
        store_path = '../data_warehouse/local_data/'
        fileparm_name = store_path + fileparm_name
        file_name = ''.join(file_name.split('.')[:-1])+'.bin'
        self.data.tofile(store_path+file_name)
        with open(fileparm_name,'ab') as writer:
            infors = file_name + '/t' + str(self.data.shape) + '/t' + str(np.dtype(self.data)) + '/n'
            writer.write(infors)



