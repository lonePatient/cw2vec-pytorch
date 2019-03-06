#encoding:utf-8
import os
import warnings
from pycw2vec.io.data_transformer import DataTransformer
from pycw2vec.config.cw2vec_config import configs as config
warnings.filterwarnings("ignore")

def main():
    data_transformer = DataTransformer(embedding_path = config['save_gensim_vector_path'],
                                       stroke2word_path = config['save_idx2word_path'])
    data_transformer.get_similar_words(word = '中国', w_num = 10)
    data_transformer.get_similar_words(word = '男人', w_num = 10)
    data_transformer.get_similar_words(word = '女人', w_num = 10)

if __name__ =="__main__":
    main()
