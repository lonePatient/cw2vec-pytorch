#encoding:utf-8
import re
class Preprocessor(object):
    def __init__(self,min_len = 2,stopwords_path = None):
        self.min_len = min_len
        self.stopwords_path = stopwords_path
        self.reset()

    def reset(self):
        if self.stopwords_path:
            with open(self.stopwords_path,'r') as fr:
                self.stopwords = {}
                for line in fr:
                    word = line.strip(' ').strip('\n')
                    self.stopwords[word] = 1

    # 去除长度小于min_len的文本
    def clean_length(self,x):
        if len(x.split(" ")) >= self.min_len:
            return x

    #去除停用词
    def remove_stopword(self,sentence):
        words = sentence.split()
        x = [word for word in words if word not in self.stopwords]
        return " ".join(x)

    def remove_numbers(self,sentence):
        words = sentence.split()
        x = [re.sub('\d+','',word) for word in words]
        return ' '.join([w for w in x if w !=''])

    def __call__(self, sentence):
        # TorchText returns a list of words instead of a normal sentence.
        # First, create the sentence again. Then, do preprocess. Finally, return the preprocessed sentence as list
        # of words
        x = sentence
        if self.stopwords_path:
            x = self.remove_stopword(x)
        x = self.clean_length(x)
        x = self.remove_numbers(x)

        return x
