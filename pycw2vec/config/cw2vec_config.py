#encoding:utf-8
from os import path
import multiprocessing
BASE_DIR = 'pycw2vec'

configs = {
    'data_path': path.sep.join([BASE_DIR,'dataset/raw/zhihu.txt']),   # 总的数据，一般是将train和test何在一起构建语料库
    'model_save_path': path.sep.join([BASE_DIR,'output/checkpoints/cw2vec.pth']),
    'vocab_path': path.sep.join([BASE_DIR,'dataset/processed/vocab.pkl']), # 语料数据
    'ngram_vocab_path': path.sep.join([BASE_DIR,'dataset/processed/ngram_vocab.pkl']),
    'word_ngrams_path': path.sep.join([BASE_DIR,'dataset/processed/word_ngrams.pkl']),
    'word_embedding_path': path.sep.join([BASE_DIR,'output/embedding/cw2vec.bin']),
    'all_embedding_path': path.sep.join([BASE_DIR,'output/embedding/all_cw2vec.bin']), # 包含ngram
    'char_to_stroke_path':path.sep.join([BASE_DIR,'dataset/processed/char_to_stroke.pkl']), # 字符转化为strokes

    'log_dir': path.sep.join([BASE_DIR, 'output/log']),           # 模型运行日志
    'figure_dir': path.sep.join([BASE_DIR, 'output/figure']),     # 图形保存路径
    'stopword_path': path.sep.join([BASE_DIR,'dataset/raw/stopwords.txt']),
    'stroke_path':path.sep.join([BASE_DIR,'dataset/raw/strokes.txt']),
    'vocab_size':300000,
    'embedding_dim':100,
    'epochs':6,
    'batch_size':64,
    'window_size':5,
    'negative_sample_num':5,
    'n_gpus':[],
    'min_freq':5,
    'max_seq_len':70,
    'sample':1e-3,

    'num_workers':multiprocessing.cpu_count(),
    'learning_rate':0.0025,
    'weight_decay':5e-4,
    'lr_min':0.00001,
    'lr_patience': 3, # number of epochs with no improvement after which learning rate will be reduced.
    'mode': 'min',    # one of {min, max}
    'monitor': 'loss',  # 计算指标
}
