# encoding:utf-8
import argparse
from pycw2vec.utils.logginger import init_logger
from pycw2vec.config.cw2vec_config import configs as config
from pycw2vec.model.nn import gensim_fasttext
from pycw2vec.preprocessing.preprocessor import Preprocessor
from pycw2vec.io.dataset import DataLoader

def main():
    logger = init_logger(log_name='gensim_fasttext', log_dir=config['log_dir'])
    logger.info('load data from disk' )
    data_loader = DataLoader(skip_header  = False,
                            shuffle      = True,
                            strokes_path = config['stroke_path'],
                            negative_num = config['negative_sample_num'],
                            batch_size   = config['batch_size'],
                            window_size  = config['window_size'],
                            data_path    = config['data_path'],
                            vocab_path   = config['vocab_path'],
                            vocab_size   = config['vocab_size'],
                            min_freq     = config['min_freq'],
                            max_seq_len  = config['max_seq_len'],
                            seed         = args['seed'],
                            sample       = config['sample'],
                            ngram_vocab_path = config['ngram_vocab_path'],
                            char_to_stroke_path = config['char_to_stroke_path'],
                            stroke2idx = config['stroke2idx'])
    # 加载数据
    examples,stroke2word = data_loader.generator_gensim_data(idx2sentence_path = config['save_sentence2idx_path'],
                                                            idx2word_path = config['save_idx2word_path'])

    logger.info("initializing emnedding model")
    model = gensim_fasttext.FastText(sg = 1,
                                    iter = 15,
                                    logger = logger,
                                    size = config['embedding_dim'],
                                    window = config['window_size'],
                                    min_count = config['min_freq'],
                                    num_workers = config['num_workers'],
                                    seed = args['seed'],
                                    word_ngrams = config['word_ngrams'])

    model.train_fasttext([document.split(" ") for document in examples])

    model.save(save_model_path = config['save_gensim_model_path'],
               save_vectors_path = config['save_gensim_vector_path'])

if __name__ == "__main__":

    ap = argparse.ArgumentParser(description='PyTorch model training')
    ap.add_argument('-s',
                    '--seed',
                    default=2018,
                    type=str,
                    help='Seed for initializing training.')
    args = vars(ap.parse_args())
    main()