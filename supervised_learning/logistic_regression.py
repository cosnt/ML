import numpy as np

class LogRegres(object):
    def __init__(self, data, parm):
        self.data = data
        self.parm = parm
        self.theta = None
        self.cost = []

    def fit():
        feat_num, classfy_num = data.train_set.shape[1], data.train_label.shape[1]
        self.theta = np.matlib.zeros((feat_num, classfy_num))
        for step in range(self.parm.max_step):
            train_data_batch, train_label_batch = get_batch(self.data.train_data, self.data.train_label, batch_size)
            for index in range(classfy_num):
                self.theta[index,:] +=  grad;
            cost.append(loss)

    def predict():
        pass

def main():
    file_name = '/mnt/'
    data, parm = Data(), OptimizationParm()
    model = LogRegres(data, parm)
    model.fit()
    model.predict()

if __name__ == '__mian__':
    main()    
