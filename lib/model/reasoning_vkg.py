import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.module.language_model import WordEmbedding, QuestionEmbedding
from lib.module.classifier import SimpleClassifier
from lib.module.reasoning_unit import ReasoningUnit
from block import fusions

class RelationVKG(nn.Module):

    def __init__(self, cfg):
        super(RelationVKG, self).__init__()

        q_dim = cfg['rnn_dim']*2 if cfg['rnn_bidirection'] else cfg['rnn_dim']
        self.w_emb = WordEmbedding(cfg['n_vocab'], cfg['word_embedding_dim'])
        self.w_emb.init_embedding(cfg['word_dic_file'], cfg['embedding_file'])
        self.q_emb = QuestionEmbedding(cfg['word_embedding_dim'], cfg['rnn_dim'], cfg['rnn_layer'],
                                      cfg['rnn_type'], keep_seq=True, bidirectional=cfg['rnn_bidirection'])
        self.reasoning_net = ReasoningUnit(cfg['v_dim'], q_dim, cfg['rel_dim'], cfg['node_att_hid_dim'],
                                           gat_att_hid_dim=cfg['gat_att_hid_dim'], gat_out_dim=cfg['v_dim'],
                                           gat_n_att=cfg['gat_n_att'],
                                           gat_multi_head_type="concat",
                                           que_self_att_enable=cfg['ques_self_att_enable'],
                                           node_att_enable=cfg['node_att_enable'], gat_enable=cfg['gat_enable'],
                                           spatial_feature_enable=cfg['spatial_feature_enable'], recurrent=cfg['recurrent'],
                                           dropout=cfg['dropout'], wn=cfg['wn'])
        if cfg['fuse_type'] == 'LinearSum':
            self.fuse_net = fusions.LinearSum([cfg['v_dim'], q_dim], cfg['fused_dim'], dropout_input=cfg['dropout'])
        if cfg['fuse_type'] == 'MFB':
            self.fuse_net = fusions.MFB([cfg['v_dim'], q_dim], cfg['fused_dim'], dropout_input=cfg['dropout'])
        if cfg['fuse_type'] == 'MLB':
            self.fuse_net = fusions.MLB([cfg['v_dim'], q_dim], cfg['fused_dim'], mm_dim=2*cfg['fused_dim'], dropout_input=cfg['dropout'])
        if cfg['fuse_type'] == 'BLOCK':
            self.fuse_net = fusions.Block([cfg['v_dim'], q_dim], cfg['fused_dim'], mm_dim=2*cfg['fused_dim'], dropout_input=cfg['dropout'])
        self.classifier = SimpleClassifier(cfg['fused_dim'], cfg['classifier_hid_dim'], cfg['classes'], 0.5)

    def forward(self, ent_emb, spatial, bbox, q_tokens, q_len):

        w_emb = self.w_emb(q_tokens)
        q_emb = self.q_emb(w_emb, q_len)

        ent_emb, q_emb, attention = self.reasoning_net(ent_emb, spatial, bbox, q_emb)

        fused_emb = self.fuse_net([ent_emb.sum(1), q_emb.sum(1)])

        logits = self.classifier(fused_emb)

        return logits, attention
