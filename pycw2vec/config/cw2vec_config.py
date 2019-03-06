#encoding:utf-8
from pathlib import Path
import multiprocessing

BASE_DIR = Path('pycw2vec')

configs = {
    'data_path': BASE_DIR / 'dataset/raw/zhihu.txt',
    'model_save_path': BASE_DIR / 'output/checkpoints/cw2vec.pth',
    'vocab_path': BASE_DIR / 'dataset/processed/vocab.pkl', # 语料数据
    'ngram_vocab_path': BASE_DIR / 'dataset/processed/ngram_vocab.pkl',
    'word_ngrams_path': BASE_DIR / 'dataset/processed/word_ngrams.pkl',
    'word_embedding_path': BASE_DIR / 'output/embedding/cw2vec.bin',
    'all_embedding_path': BASE_DIR / 'output/embedding/all_cw2vec.bin', #含ngram
    'char_to_stroke_path': BASE_DIR /'dataset/processed/char_to_stroke.pkl', #字符转化为strokes

    'save_gensim_model_path':BASE_DIR / 'output/checkpoints/gensim_cw2vec.bin',
    'save_gensim_vector_path':BASE_DIR / 'output/embedding/gensim_word_vector.bin',
    'save_sentence2idx_path': BASE_DIR / 'dataset/processed/sentence2idx.pkl',
    'save_idx2word_path':BASE_DIR /'dataset/processed/idx2word.pkl',

    'log_dir': BASE_DIR / 'output/log',           # 模型运行日志
    'figure_dir': BASE_DIR / 'output/figure',     # 图形保存路径
    'stopword_path': BASE_DIR / 'dataset/raw/stopwords.txt',
    'stroke_path':BASE_DIR / 'dataset/raw/strokes.txt',

    'vocab_size':300000,
    'embedding_dim':50,
    'epochs':6,
    'batch_size':64,
    'window_size':5,
    'negative_sample_num':5,
    'n_gpus':[1],
    'min_freq':1,
    'max_seq_len':70,
    'sample':1e-3,
    'word_ngrams':1,  # If 1, uses enriches word vectors with subword(n-grams) information.
                      # If 0, this is equivalent to :class:`~gensim.models.word2vec.Word2Vec`.

    'num_workers':multiprocessing.cpu_count(),
    'learning_rate':0.0025,
    'weight_decay':5e-4,
    'lr_min':0.00001,
    'lr_patience': 3, # number of epochs with no improvement after which learning rate will be reduced.
    'mode': 'min',    # one of {min, max}
    'monitor': 'loss',  # 计算指标

    'stroke2idx':{ # 按照原始论文 其他默认为5，这里就不列出
         '横':1,
         '提':1,
         '竖':2,
         '竖钩':2,
         '撇':3,
         '捺':4,
         '点':4
         },

    # 'stroke2idx':{ # 这里把所有笔画做一次实验进行对比
    #          '捺':1,
    #          '提':2,
    #          '撇':3,
    #          '撇折':4,
    #          '撇点':5,
    #          '斜钩':6,
    #          '横':7,
    #          '横折':8,
    #          '横折弯钩':9,
    #          '横折折':10,
    #          '横折折/横折弯':11,
    #          '横折折折':12,
    #          '横折折折钩':13,
    #          '横折折撇':14,
    #          '横折提':15,
    #          '横折竖钩':16,
    #          '横撇':17,
    #          '点':18,
    #          '竖':19,
    #          '竖弯横钩':20,
    #          '竖折':21,
    #          '竖折折钩':22,
    #          '竖折撇':23,
    #          '竖提':24,
    #          '竖钩':25,
    #         '弯钩':26,
    #         },

}
