import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from torch.nn.utils.weight_norm import weight_norm
from lib.module.language_model import WordEmbedding, QuestionEmbedding
from lib.module.classifier import SimpleClassifier
from lib.module.fc import FCNet
from lib.module.rn import RN
from block import fusions
class RNModel(nn.Module):
    def __init__(self,cfg):
        super(RNModel, self).__init__()
        self.glimpse = 1
        q_dim = cfg['rnn_dim'] * 2 if cfg['rnn_bidirection'] else cfg['rnn_dim']
        self.w_emb = WordEmbedding(cfg['n_vocab'], cfg['word_embedding_dim'])
        self.w_emb.init_embedding(cfg['word_dic_file'], cfg['embedding_file'])
        self.q_emb = QuestionEmbedding(cfg['word_embedding_dim'], cfg['rnn_dim'], cfg['rnn_layer'],
                                      cfg['rnn_type'], keep_seq=False, bidirectional=cfg['rnn_bidirection'])
        self.v_att = StackedAttention(1, cfg['v_dim'], q_dim, cfg['fused_dim']) if cfg['att_enable'] else None
        self.rn = RN(cfg['v_dim'], q_dim, subspace_dim=cfg['rn_sub_dim'], relation_glimpse=1, pe_enable=cfg['pe_enable'], ksize=cfg['ksize'])
        self.att_v_net = FCNet([cfg['v_dim'], cfg['fused_dim']]) if cfg['att_enable'] else None
        self.rn_v_net = FCNet([cfg['v_dim'], cfg['fused_dim']])
        self.q_net = FCNet([q_dim, cfg['fused_dim']])
        self.classifier = SimpleClassifier(cfg['fused_dim'], cfg['classifier_hid_dim'], cfg['classes'], 0.5)


    def forward(self, v, b, q, q_len):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb, q_len) # [batch, q_dim]

        v_emb = v
        if self.v_att is not None:
            attentive_v, att = self.v_att(v, q_emb)
        #    attentive_v = nn.functional.normalize(attentive_v, dim=2)
            att = att.squeeze(2).unsqueeze(1)
        else:
            attentive_v = None
            att = None


        if self.rn is not None:
            rn_map, rn_v = self.rn(v, q_emb, b)
        #    rn_v = nn.functional.normalize(rn_v, dim=2)xdcddd
        else:
            rn_map = None
            rn_v = None

        if attentive_v is not None and rn_v is not None:
            v_repr = self.att_v_net(attentive_v.sum(1)) * self.rn_v_net(rn_v.sum(1))
        elif attentive_v is not None and rn_v is None:
            v_repr = self.att_v_net(attentive_v.sum(1))
        elif attentive_v is None and rn_v is not None:
            v_repr = self.rn_v_net(rn_v.sum(1))
        else:
            v_repr = self.v_net(v_emb.sum(1))
        
        q_repr = self.q_net(q_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits, att

class BilinearAttentionLayer(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(BilinearAttentionLayer, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid], dropout)
        self.q_proj = FCNet([q_dim, num_hid], dropout)
        self.h_mat = nn.Parameter(torch.Tensor(1, 1, num_hid).normal_())
        self.h_bias = nn.Parameter(torch.Tensor(1, 1, 1).normal_())

    def forward(self, v, q, prev_logits=None):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        if prev_logits is None:
            logits = self.logits(v, q) #[batch, k, 1]
        else:
            logits = self.logits(v, q) + prev_logits
        w = nn.functional.softmax(logits, 1) #[batch, k, 1]
        v = w * v  #[batch, k, vdim]
        return v, logits, w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, num_hid]
        q_proj = self.q_proj(q).unsqueeze(1).transpose(1, 2)# [batch, num_hid, 1]
        v_proj = v_proj * self.h_mat
        logits = torch.matmul(v_proj, q_proj) + self.h_bias #[batch, k, 1]
        return logits

class StackedAttention(nn.Module):
    def __init__(self, stacked, v_dim, q_dim, num_hid, dropout=0.2):
        super(StackedAttention, self).__init__()
        self.stacked = stacked
        attLayers = []
        for i in range(stacked):
            attLayers.append(BilinearAttentionLayer(v_dim, q_dim, num_hid * 3, dropout))
        self.attLayers = nn.ModuleList(attLayers)

    def forward(self, v, q):
        prev_att_logits = None
        attentive_v = None
        att = None
        for s in range(self.stacked):
            attentive_v, att_logits, att = self.attLayers[s].forward(v, q, prev_att_logits)
            prev_att_logits = att_logits
            if s < self.stacked-1:
                v = attentive_v + v
        return attentive_v, att
