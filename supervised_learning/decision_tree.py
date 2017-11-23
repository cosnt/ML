import numpy as np
import math

class OptimizationParameters(object):
    pass
#矩阵与
#待修理，整合数据结构
class DecTree(object):
    def __init__(self,data, parm):
        self.data = data
        self.parm = parm
        self.result = {}
        self.feats = {}

    def _tree_generate(self, data, attributes):
        if len(set(data[:,-1])) == 1:
            return data[0,-1]
        elif attributes is None or self._is_all_same(data, attributes):
            return self._get_most_label(data[:,-1])
        else:
            attribute = self._get_best_attribute(attributes)
            for value in self.feats[attribute]:
                data= self._get_data(data, attribute, value)
                if data is None:
                    return self._get_most_label(data)
                else:
                    self._tree_generate(data, attributes.remove(attribute))

    def _is_all_same(self, data, attributes):
        for attribute in attributes:
            if len(set(data[:,attribute])) != 1:
                return False
        return True

    def _get_most_label(self,data):
        label = data[:,-1]
        return max(set(label), key=label.count)

    def _get_best_attribute(self,attributes):
        pass

    def _get_data(self,data, attribute, value):
        index = data[:,attribute]==value
        return data[:,index]

    def _information_entropy(self):
        pass

    def _information_gain(self):
        pass

    def _gini_index(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    def display_tree(self):
        pass

def preset_parm():
    pass

def main():
    parm = preset_parm()
    dec_tree = DecTree(parm)
    dec_tree.predict()

if __name__ == '__main__':
    main()
