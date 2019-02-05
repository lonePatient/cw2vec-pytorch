#encoding:utf-8
import math
import random
import numpy as np
import operator
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import Counter
from ..utils.utils import pkl_write

class DataLoader(Dataset):
    def __init__(self,
                 seed,
                 shuffle,
                 sample,
                 data_path,
                 window_size,
                 skip_header,
                 negative_num,
                 min_freq,
                 vocab_size,
                 vocab_path,
                 max_seq_len,
                 batch_size,
                 strokes_path,
                 ngram_vocab_path,
                 char_to_stroke_path,
                 processing = None
                 ):

        self.seed    = seed
        self.shuffle = shuffle
        self.window_size  = window_size
        self.skip_header  = skip_header
        self.negative_num = negative_num
        self.min_freq    = min_freq
        self.skip_header = skip_header
        self.vocab_size  = vocab_size
        self.batch_size  = batch_size
        self.processing  = processing
        self.max_seq_len = max_seq_len
        self.sample      = sample
        self.data_path   = data_path
        self.vocab_path  = vocab_path
        self.ngram_vocab_path = ngram_vocab_path
        self.strokes_path     = strokes_path
        self.char_to_stroke_path = char_to_stroke_path
        self.random_s = np.random.RandomState(self.seed)

        self.build_examples()              # 读取所有数据集，一行一行
        self.build_vocab()                 # 建立语料困
        self.build_negative_sample_table() # 根据词频构建负采样
        self.build_strokes_mapping()       # 构建笔画库
        self.build_ngram_vocab()           # 构建语料库中词对应的笔画
        self.subsampling()                 # 下采样
        self.build_word_features()

    def reserve_ratio(self,p,total):
        tmp_p = (math.sqrt( p / self.sample) + 1 ) * self.sample / p
        if tmp_p >1:
            tmp_p = 1
        return tmp_p * total

    # 分割数据
    def split_sent(self,line):
        res = line.split()
        return res

    # 将词转化为id
    def word_to_id(self,word, vocab):
        return vocab[word][0] if word in vocab else vocab['<unk>'][0]

    # 读取数据，并进行预处理,将每一个句子分割成词的列表
    def build_examples(self):
        self.examples = []
        with open(self.data_path, 'r') as fr:
            for i, line in tqdm(enumerate(fr), desc='read data and processing'):
                # 数据首行为列名
                if i == 0 and self.skip_header:
                    continue
                line = line.strip("\n")
                if self.processing:
                    line = self.processing(line)
                if line:
                    self.examples.append(self.split_sent(line))

    # 数据采样，降低高频词的出现
    def subsampling(self,total = 2 ** 32):
        pow_frequency = np.array(list(self.word_frequency.values()))
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        delete_int = [self.reserve_ratio(p,total = total) for p in ratio]

        self.train_examples = []
        for example in self.examples:
            words = [self.vocab[word] for word in example if
                           word in self.vocab and delete_int[self.vocab[word]] >= random.random() * total]
            if len(words) > 0:
                self.train_examples.append(words)
        del self.examples

    # 建立语料库（词为主）
    def build_vocab(self):
        count = Counter()
        for words in tqdm(self.examples,desc = 'build vocab'):
            count.update(words)
        count = {k: v for k, v in count.items()}
        count = sorted(count.items(), key=operator.itemgetter(1),reverse=True)
        all_words = [(w[0],w[1]) for w in count if w[1] >= self.min_freq]
        if self.vocab_size:
            all_words = all_words[:self.vocab_size]
        all_words =  all_words + [('<unk>',0)]
        word2id = {k: (i,v) for i,(k, v) in zip(range(0, len(all_words)),all_words)}
        self.word_frequency = {tu[0]: tu[1] for word, tu in word2id.items()}
        self.vocab = {word: tu[0] for word, tu in word2id.items()}
        pkl_write(data = word2id,filename=self.vocab_path)

    # 构建笔画语料库,每一个中文对应的笔画信息
    def build_strokes_mapping(self):
        strok_to_id = {'横':'1','提':'1','竖':'2','竖钩':'2','撇':'3','捺':'4','点':'4'}
        self.char_to_stroke = {}
        with open(self.strokes_path,'r',encoding='utf-8') as fr:
            for line in fr:
                lines = line.strip().split(":")
                if len(lines) == 2:
                    arr = lines[1].split(",")
                    strokes = [strok_to_id.get(stroke,'5') for stroke in arr]
                    self.char_to_stroke[lines[0]] = ''.join(strokes)
        pkl_write(data= self.char_to_stroke,filename=self.char_to_stroke_path)

    # 根据词频构建负样本
    def build_negative_sample_table(self):
        self.negative_sample_table = []
        sample_table_size = 1e8
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * sample_table_size)
        for wid, c in enumerate(count):
            self.negative_sample_table += [wid] * int(c)
        self.negative_sample_table = np.array(self.negative_sample_table)

    # 构建一个词的ngram字符特征
    def char_ngram_generator(self,word,n1 = 3,n2 = 5):
        z = []
        text = ''
        for char in list(word):
            strokes = self.char_to_stroke.get(char,None)
            if strokes:
                text += strokes
        if text == '':
            return []
        for k in range(n1,n2+1):
            z.append([text[i:i+k] for i in range(len(text) - k + 1)])
        z = ['0'+ngram for ngrams in z for ngram in ngrams]
        return z

    # 建立ngram语料库以及每一个词对应的ngram
    def build_ngram_vocab(self):
        self.word_ngrams = {}
        count = Counter()
        for word,word_id in self.vocab.items():
            ngram_feature = self.char_ngram_generator(word)
            count.update(ngram_feature)
        count = {k: v for k, v in count.items()}
        count = sorted(count.items(), key=operator.itemgetter(1))
        all_ngrams = [(w[0],w[1]) for w in count]
        self.ngram2id = {k: (i+len(self.vocab),v) for i, (k,v) in zip(range(0, len(all_ngrams)), all_ngrams)}
        self.all_vocab = dict(self.vocab,**self.ngram2id)
        pkl_write(data=self.ngram2id, filename=self.ngram_vocab_path)

    # 构建词与ngram特征映射
    def build_word_features(self):
        self.word_ngrams = {}
        for word, word_id in tqdm(self.vocab.items(),desc = 'building word mapping feature'):
            ngram_feature = self.char_ngram_generator(word)
            if len(ngram_feature) == 0:
                continue
            self.word_ngrams[word_id] = (word_id,)
            self.word_ngrams[word_id] += tuple([self.ngram2id[k][0] for k in ngram_feature])

    # 负样本
    def get_neg_word(self,u):
        neg_v = []
        while len(neg_v) < self.negative_num:
            n_w = np.random.choice(self.negative_sample_table,size = self.negative_num).tolist()[0]
            if n_w != u:
                neg_v.append(n_w)
        return neg_v

    # 构建skip_gram对应的列表（这里还是词的两两组合）,也可以转换为二分类来构造向量
    # 那么window_size内的词label为1，负样本label为0
    # 构建skip gram模型样本
    def make_iter(self):
        for example in self.train_examples:
            if len(example) < 2:
                continue
            for i,w in enumerate(example):
                if self.word_ngrams.get(w,None) == None:
                    continue
                reduced_window = self.random_s.randint(self.window_size)
                words_num = len(example)
                window_start = max(0, i - self.window_size + reduced_window)
                window_end = min(words_num, i + self.window_size + 1 - reduced_window)
                pos_v = [example[j] for j in range(window_start, window_end) if j != i]
                pos_u = [self.word_ngrams[w]] * len(pos_v)
                neg_u = [c for c in pos_u for _ in range(self.negative_num)]
                neg_v = [nv for v in pos_v for nv in self.get_neg_word(v)]
                yield pos_u,pos_v,neg_u,neg_v

