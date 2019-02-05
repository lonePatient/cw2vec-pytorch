#encoding:utf-8
import argparse
import torch
import warnings
from torch import optim
from pycw2vec.train.trainer import Trainer
from pycw2vec.io.dataset import DataLoader
from pycw2vec.model.nn.skipgram import SkipGram
from pycw2vec.utils.logginger import init_logger
from pycw2vec.utils.utils import seed_everything
from pycw2vec.config.cw2vec_config import configs as config
from pyword2vec.callback.lrscheduler import StepLr
from pycw2vec.callback.trainingmonitor import TrainingMonitor
warnings.filterwarnings("ignore")

# 主函数
def main():
    arch = 'cw2vec'
    logger = init_logger(log_name=arch, log_dir=config['log_dir'])
    logger.info("seed is %d"%args['seed'])
    seed_everything(seed = args['seed'])

    #**************************** 加载数据集 ****************************
    logger.info('starting load train data from disk')
    train_dataset   = DataLoader(skip_header  = False,
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
                                    char_to_stroke_path = config['char_to_stroke_path']
                                    )
    # **************************** 模型和优化器 ***********************
    logger.info("initializing model")
    model = SkipGram(embedding_dim = config['embedding_dim'],vocab_size = len(train_dataset.all_vocab))
    optimizer = optim.Adam(params = model.parameters(),lr = config['learning_rate'])

    # **************************** callbacks ***********************
    logger.info("initializing callbacks")

    # 监控训练过程
    train_monitor = TrainingMonitor(fig_dir  = config['figure_dir'],
                                    json_dir = config['log_dir'],
                                    arch     = arch)
    # 学习率机制
    lr_scheduler = StepLr(optimizer=optimizer,
                          init_lr=config['learning_rate'],
                          epochs=config['epochs'])

    # **************************** training model ***********************
    logger.info('training model....')
    trainer = Trainer(model            = model,
                      train_data       = train_dataset,
                      optimizer        = optimizer,
                      epochs           = config['epochs'],
                      logger           = logger,
                      training_monitor = train_monitor,
                      lr_scheduler     = lr_scheduler,
                      n_gpu            = config['n_gpus'],
                      vocab            = train_dataset.vocab,
                      all_vocab        = train_dataset.all_vocab,
                      model_save_path=config['model_save_path'],
                      vector_save_path=config['word_embedding_path'],
                      all_vector_save_path=config['all_embedding_path']
                      )
    # 查看模型结构
    trainer.summary()
    # 拟合模型
    trainer.train()
    # 释放显存
    if len(config['n_gpus']) > 0:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='PyTorch model training')
    ap.add_argument('-s',
                    '--seed',
                    default=1024,
                    type = int,
                    help = 'Seed for initializing training.')

    ap.add_argument('-r',
                    '--resume',
                    default = False,
                    type = bool,
                    help = 'Choose whether resume checkpoint model')
    args = vars(ap.parse_args())
    main()


