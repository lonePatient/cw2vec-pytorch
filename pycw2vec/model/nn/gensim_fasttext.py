#encoding:utf-8
from gensim.models import fasttext

class FastText():
    def __init__(self,
                 size,
                 sg,
                 iter,
                 seed,
                 logger,
                 window,
                 num_workers,
                 word_ngrams,
                 min_count):

        self.sg = sg
        self.size = size
        self.seed = seed
        self.iter = iter
        self.window = window
        self.logger = logger
        self.min_count = min_count
        self.workers = num_workers
        self.word_ngrams = word_ngrams

    def train_fasttext(self, data):
        self.logger.info('train fasttext....')
        self.logger.info(f'word vector size is: {self.size}')

        self.model = fasttext.FastText(data,
                                       sg = self.sg,
                                       iter = self.iter,
                                       seed = self.seed,
                                       size = self.size,
                                       window = self.window,
                                       workers = self.workers,
                                       min_count = self.min_count,
                                       word_ngrams = self.word_ngrams)

    def save(self,save_model_path,save_vectors_path):
        self.logger.info('saving fasttext model ....')
        self.model.save(str(save_model_path))

        self.logger.info('saving word vectors ....')
        with open(str(save_vectors_path),'w') as fw:
            for word in self.model.wv.vocab:
                vector = self.model[word]
                fw.write(str(word) + ' ' + ' '.join(map(str, vector)) + '\n')
