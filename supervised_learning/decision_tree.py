import numpy as np
import math

class OptimizationParameters(object):
    pass

class DecTree(object):
    def __init__(self,data, parm):
        self.data = data
        self.parm = parm

    def fit(self):
        pass
    def predict(self):
        pass
def preset_parm():
    pass

def main():
    parm = preset_parm()
    dec_tree = DecTree(parm)
    dec_tree.predict()

if __name__ == '__main__':
    main()
