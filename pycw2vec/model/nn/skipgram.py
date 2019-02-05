#encoding:Utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGram(torch.nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(SkipGram, self).__init__()
        initrange = 0.5/ embedding_dim
        self.u_embedding_matrix = nn.Embedding(vocab_size,embedding_dim)
        self.v_embedding_matrix = nn.Embedding(vocab_size,embedding_dim)
        self.u_embedding_matrix.weight.data.uniform_(-initrange,initrange)
        self.v_embedding_matrix.weight.data.uniform_(-0,0) # 保存的信息

    def forward(self, pos_u, pos_v,neg_u, neg_v):
        embed_pos_u = self.u_embedding_matrix(pos_u)
        embed_pos_v = self.v_embedding_matrix(pos_v)
        pos_input_word  = torch.sum(embed_pos_u, dim=1) / len(pos_u[0][1:])
        score = torch.mul(pos_input_word, embed_pos_v)
        score = torch.sum(score, dim=1)
        log_target = F.logsigmoid(score).squeeze()

        embed_neg_u = self.u_embedding_matrix(neg_u)
        embed_neg_v = self.v_embedding_matrix(neg_v)

        neg_input_word  = torch.sum(embed_neg_u, dim=1) / len(pos_u[0][1:])
        neg_score = torch.mul(neg_input_word,embed_neg_v)
        neg_score = torch.sum(neg_score, dim=1)
        sum_log_sampled = F.logsigmoid(-1 * neg_score).squeeze()

        loss = log_target.sum() + sum_log_sampled.sum()
        loss = -1 * loss
        return loss / len(pos_v)
