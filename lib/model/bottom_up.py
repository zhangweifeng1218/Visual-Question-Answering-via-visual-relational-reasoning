import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.module.language_model import WordEmbedding, QuestionEmbedding
from lib.module.classifier import SimpleClassifier
from lib.module.fc import FCNet
from torch.nn.utils.weight_norm import weight_norm
from block import fusions

class BottomUp(nn.Module):

    def __init__(self, cfg):
        super(BottomUp, self).__init__()
        self.cfg = cfg
        q_dim = cfg['rnn_dim']*2 if cfg['rnn_bidirection'] else cfg['rnn_dim']
        self.w_emb = WordEmbedding(cfg['n_vocab'], cfg['word_embedding_dim'])
        self.w_emb.init_embedding(cfg['word_dic_file'], cfg['embedding_file'])
        self.q_emb = QuestionEmbedding(cfg['word_embedding_dim'], cfg['rnn_dim'], cfg['rnn_layer'],
                                      cfg['rnn_type'], keep_seq=False, bidirectional=cfg['rnn_bidirection'])
        self.v_att = NewAttention(cfg['v_dim'], q_dim, cfg['fused_dim'])
        if cfg['fuse_type'] == 'LinearSum':
            self.fuse_net = fusions.LinearSum([cfg['v_dim'], q_dim], cfg['fused_dim'], dropout_input=cfg['dropout'])
        if cfg['fuse_type'] == 'MFB':
            self.fuse_net = fusions.MFB([cfg['v_dim'], q_dim], cfg['fused_dim'], mm_dim=1000, factor=5, dropout_input=cfg['dropout'])
        if cfg['fuse_type'] == 'MLB':
            self.fuse_net = fusions.MLB([cfg['v_dim'], q_dim], cfg['fused_dim'], mm_dim=2*cfg['fused_dim'], dropout_input=cfg['dropout'])
        if cfg['fuse_type'] == 'MFH':
            self.fuse_net = fusions.MFH([cfg['v_dim'], q_dim], cfg['fused_dim'], mm_dim=1000, factor=5, dropout_input=cfg['dropout'])
        if cfg['fuse_type'] == 'MCB':
            from compact_bilinear_pooling import CompactBilinearPooling
            self.fuse_net = CompactBilinearPooling(cfg['v_dim'], q_dim, cfg['fused_dim'])
           # self.fuse_net = fusions.MCB([cfg['v_dim'], q_dim], cfg['fused_dim'], dropout_output=cfg['dropout'])
        self.classifier = SimpleClassifier(cfg['fused_dim'], cfg['classifier_hid_dim'], cfg['classes'], 0.5)

    def forward(self, v_emb, spatial, bbox, q_tokens, q_len):

        w_emb = self.w_emb(q_tokens)
        q_emb = self.q_emb(w_emb, q_len)

        att = self.v_att(v_emb, q_emb)
        v_emb = (att * v_emb).sum(1)

        if self.cfg['fuse_type'] == 'MCB':
            fused_emb = self.fuse_net(v_emb, q_emb)
        else:
            fused_emb = self.fuse_net([v_emb, q_emb])

        logits = self.classifier(fused_emb)

        return logits


class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits