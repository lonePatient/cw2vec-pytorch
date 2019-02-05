#encoding:utf-8
import os
import numpy as np
from ..utils.utils import pkl_read
from sklearn.metrics.pairwise import cosine_similarity

class DataTransformer(object):
    def __init__(self,
                 vocab_path,
                 embedding_path):
        self.vocab_path = vocab_path
        self.embedding_path = embedding_path
        self.reset()

    def reset(self):
        if os.path.isfile(self.vocab_path):
            self.vocab = pkl_read(self.vocab_path)
        else:
            raise FileNotFoundError("vocab file not found")
        self.load_embedding()

    def build_embedding_matrix(self,emb_mean = None,emb_std = None):
        '''
        构建词向量权重矩阵
        :param embedding_path:
        :param embedding_dim:
        :param oov_type:
        :return:
        '''
        embeddings_index = self.load_embedding()
        all_embs = np.stack((embeddings_index.values()))
        if emb_mean is None or emb_std is None:
            emb_mean = all_embs.mean()
            emb_std  = all_embs.std()
        embed_size = all_embs.shape[1]
        nb_words = len(self.vocab)
        # 这里我们简单使用正态分布产生随机值
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        for word, id in self.vocab.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[id] = embedding_vector
        return embedding_matrix

    # 加载词向量矩阵
    def load_embedding(self, ):
        print(" load emebedding weights")
        self.embeddings_index = {}
        self.words = []
        self.vectors = []
        f = open(self.embedding_path, 'r',encoding = 'utf8')
        for line in f:
            values = line.split(' ')
            try:
                word  = values[0]
                self.words.append(word)
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs
                self.vectors.append(coefs)
            except:
                print("Error on ", values[:2])
        f.close()
        self.vectors = np.vstack(self.vectors)
        print('Total %s word vectors.' % len(self.embeddings_index))

    # 计算相似度
    def get_similar_words(self, word, w_num=10):
        if word not in self.embeddings_index:
            raise ValueError('%d not in vocab')
        current_vector = self.embeddings_index[word]
        result = cosine_similarity(current_vector.reshape(1, -1), self.vectors)
        result = np.array(result).reshape(len(self.vocab), )
        idxs = np.argsort(result)[::-1][:w_num]
        print("<<<" * 7)
        print(word)
        for i in idxs:
            print("%s : %.3f\n" % (self.words[i], result[i]))
        print(">>>" * 7)




