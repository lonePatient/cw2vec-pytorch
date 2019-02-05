#encoding:utf-8
import torch.nn.functional as F

class CrossEntropy(object):
    def __init__(self):
        super(CrossEntropy,self).__init__()
        pass
    def __call__(self, output, target):
        loss = F.cross_entropy(input=output, target=target)
        return loss