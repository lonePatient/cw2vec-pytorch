#encoding:utf-8


class Accuracy(object):
    '''
    计算准确度
    可以使用topK参数设定计算K准确度
    '''
    def __init__(self,topK):
        super(Accuracy,self).__init__()
        self.topK = topK

    def __call__(self, output, target):
        batch_size = target.size(0)
        _, pred = output.topk(self.topK, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:self.topK].view(-1).float().sum(0)
        result = correct_k / batch_size
        return result