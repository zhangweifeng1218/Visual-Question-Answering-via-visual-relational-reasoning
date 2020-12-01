import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.module.fc import FCNet

class QusSelfAttention(nn.Module):

    def __init__(self, q_dim, gamma=0.3, dropout=0.5, wn=True):
        super(QusSelfAttention, self).__init__()
        self.W = FCNet([q_dim, q_dim], wn=wn)
        self.P = FCNet([q_dim, 1], wn=wn)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.gamma = gamma

    def forward(self, q_emb, previous=None):
        '''
        :param q_emb: question feature [B, q_len, q_dim]
        :return:
        '''
        logits = self.logits(q_emb)
        attention = F.softmax(logits, dim=1)
        if previous is not None:
            attention = attention + torch.mul(previous, self.gamma)
        updated_q_emb = attention * q_emb # [ B, q_len, q_dim ]
        return updated_q_emb, attention

    def logits(self, q_emb):
        q_proj = self.dropout(self.activation(self.W(q_emb))) #[B, q_len, q_dim]
        logits = self.P(q_proj) # [B, q_len, 1]
        return logits

class NodeAttention(nn.Module):
    '''Bilinear node attention guided by question'''
    def __init__(self, v_dim, q_dim, h_dim, gamma=0.3, dropout=0.5, wn=True):
        super(NodeAttention, self).__init__()
        self.v_proj = FCNet([v_dim, h_dim], wn=wn)
        self.q_proj = FCNet([q_dim, h_dim], wn=wn)
        self.P_mat = nn.Parameter(torch.Tensor(1, 1, h_dim).normal_())
        self.P_bias = nn.Parameter(torch.Tensor(1, 1, 1).normal_())
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.gamma = gamma


    def forward(self, v_emb, q_emb, previous = None):
        '''
        :param v_emb: [B, N, v_dim]
        :param q_emb: [B, q_len, q_dim]
        :param previous: previous attention map [B, N, 1]
        :return:
        '''
        v_emb_proj = self.dropout(self.activation(self.v_proj(v_emb))) # [B, N, h_dim]
        q_emb_proj = self.dropout(self.activation(self.q_proj(q_emb))) # [B, q_len, h_dim]
        q_emb_proj = q_emb_proj.transpose(1, 2) # [B, h_dim, q_len]
        v_emb_proj = v_emb_proj * self.P_mat
        logits = torch.matmul(v_emb_proj, q_emb_proj) + self.P_bias  # [B, N, q_len]
        logits = torch.norm(logits, dim=2).unsqueeze(2) # [B, N, 1]
        attention = F.softmax(logits, dim=1) # [B, N, 1]
        if previous is not None:
            attention = attention + torch.mul(previous, self.gamma)
        updated_v = attention * v_emb
        return updated_v, attention

class RelationAttention(nn.Module):
    '''Bilinear relation attention guided by question'''
    def __init__(self, r_dim, q_dim, h_dim, gamma=0.3, dropout=0.5, wn=True):
        super(RelationAttention, self).__init__()
        self.r_proj = FCNet([r_dim, h_dim], wn=wn)
        self.q_proj = FCNet([q_dim, h_dim], wn=wn)
        self.P_mat = nn.Parameter(torch.Tensor(1, 1, 1, h_dim).normal_())
        self.P_bias = nn.Parameter(torch.Tensor(1, 1, 1, 1).normal_())
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.gamma = gamma

    def forward(self, r_emb, q_emb, previous=None):
        '''
        :param r_emb: [B, n_sbj, n_obj, r_dim]
        :param q_emb: [B, q_len, q_dim]
        :param previous: [B, n_sbj, n_obj, 1]
        :return:
        '''
        B, n_sbj, n_obj, _ = r_emb.size()
        r_emb_prj = self.dropout(self.activation(self.r_proj(r_emb)))  # [B, n_sbj, n_obj, h_dim]
        q_emb = self.dropout(self.activation(self.q_proj(q_emb)))  # [B, q_len, h_dim]
        q_emb = q_emb.transpose(1, 2)  # [B, h_dim, q_len]
        r_emb_prj = r_emb_prj * self.P_mat
        attention = torch.matmul(r_emb_prj.view(B, n_sbj*n_obj, -1), q_emb).view(B, n_sbj, n_obj, -1) + self.P_bias  # [B, n_sbj, n_obj, q_len]
        attention = torch.norm(attention, dim=3)  # [B, n_sbj, n_obj]
        attention = attention.view(B, -1)
        attention = F.softmax(attention, dim=1)  # [B, n_sbj*n_obj]
        attention = attention.view(B, n_sbj, n_obj).unsqueeze(3) # [B, n_sbj, n_obj, 1]
        if previous is not None:
            attention = attention + torch.mul(previous, self.gamma)
        r_emb = attention * r_emb
        return r_emb, attention

class QuesAttention(nn.Module):
    '''Bilinear question attention guided by vision'''
    def __init__(self, q_dim, v_dim, h_dim, gamma=0.3, dropout=0.5, wn=True):
        super(QuesAttention, self).__init__()
        self.v_proj = FCNet([v_dim, h_dim], wn=wn)
        self.q_proj = FCNet([q_dim, h_dim], wn=wn)
        self.P_mat = nn.Parameter(torch.Tensor(1, 1, h_dim).normal_())
        self.P_bias = nn.Parameter(torch.Tensor(1, 1, 1).normal_())
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.gamma = gamma


    def forward(self, q_emb, v_emb, previous = None):
        '''
        :param v_emb: [B, N, v_dim]
        :param q_emb: [B, q_len, q_dim]
        :param previous: previous attention map [B, N, 1]
        :return:
        '''
        v_emb = self.dropout(self.activation(self.v_proj(v_emb))) # [B, N, h_dim]
        q_emb = self.dropout(self.activation(self.q_proj(q_emb))) # [B, q_len, h_dim]
        v_emb = v_emb.transpose(1, 2) # [B, h_dim, N]
        q_emb = q_emb * self.P_mat
        logits = torch.matmul(q_emb, v_emb) + self.P_bias  # [B, q_len, N]
        logits = torch.norm(logits, dim=2).unsqueeze(2) # [B, q_len, 1]
        attention = F.softmax(logits, dim=1) # [B, q_len, 1]
        if previous is not None:
            attention = attention + torch.mul(previous, self.gamma)
        updated_q = attention * q_emb
        return updated_q, attention


