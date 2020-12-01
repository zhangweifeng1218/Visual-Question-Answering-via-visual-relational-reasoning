import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.module.fc import FCNet
from lib.module.attention import QusSelfAttention, NodeAttention
from lib.module.graph_attention_net import GraphAttNet, LayerNorm
import torch.nn.init as init
from block import fusions

class ReasoningUnit(nn.Module):
    '''recurrent reasoning module'''
    def __init__(self,
                 v_dim, q_dim, r_dim, node_att_hid_dim, gat_att_hid_dim, gat_out_dim, gat_n_att, gat_multi_head_type, que_self_att_enable=True,
                 node_att_enable=True, gat_enable=True, spatial_feature_enable=False, recurrent=1, gamma=0.3, dropout=0.2, wn=True):
        # gan_params:[att_hid_dim, out_dim, n_att, att_type, multi_head_type, dropout, True]
        super(ReasoningUnit, self).__init__()
        self.r_dim = r_dim
        self.ques_self_att = QusSelfAttention(q_dim, gamma, dropout, wn) if que_self_att_enable else None
        self.node_att = NodeAttention(v_dim, q_dim, node_att_hid_dim, gamma, dropout, wn) if node_att_enable else None
        self.relationEmbedding = RN(v_dim, q_dim, 2*r_dim, r_dim, pe_enable=True, dropout=dropout, wn=wn) if gat_enable else None
        self.gat = GraphAttNet(q_dim, v_dim, r_dim, gat_att_hid_dim, gat_out_dim, gat_n_att, gat_multi_head_type, dropout, wn) if gat_enable else None
        self.spatial_fea_proj = SpatialFeatureProj(v_dim) if spatial_feature_enable else None
     #   self.gate1 = SelectGate(v_dim, dropout, wn) if node_att_enable and gat_enable else None
        self.gate2 = SelectGate(v_dim, dropout, wn) if spatial_feature_enable else None
        #FCNet([v_dim, v_dim], dropout=dropout, wn=wn) if spatial_feature_enable else None  nn.Sequential([ nn.AvgPool2d(7, stride=1), FCNet([v_dim, v_dim], dropout=dropout, wn=wn)])
        #self.select_v = SelectGate(q_dim, v_dim, gan_params[0], wn)
        #self.v_layerNorm1 = LayerNorm(gan_params[1])
        #self.v_layerNorm2 = LayerNorm(gan_params[1])
        #self.q_layerNorm = LayerNorm(q_dim)
        self.recurrent = recurrent

    def forward(self, v_emb, spatial, bbox, q_emb):
        '''
        :param v_emb: [B, N, v_dim]
        :param q_emb: [B, q_len, q_dim]
        :param spatial: [B, v_dim, 7, 7]
        :return:v_emb, q_emb,same size as input
        '''
        attention = {}
        step = 0
        n_obj = v_emb.size(1)
        previous_q_self_att = None
        previous_node_att = None
        while step < self.recurrent:
            # question self attention
            if self.ques_self_att is not None:
                q_emb, previous_q_self_att = self.ques_self_att(q_emb, previous_q_self_att)
                attention['q_self_att'] = previous_q_self_att
                #q_emb = q_emb + q_emb_attended#self.q_layerNorm(q_emb + q_emb_attended)

            if self.node_att is not None:
                attentive_v_emb, previous_node_att = self.node_att(v_emb, q_emb, previous_node_att)
                attention['node_att'] = previous_node_att
                #v_emb = v_emb + attentive_v_emb
            else:
                attentive_v_emb = torch.zeros_like(v_emb)

            q_sum = q_emb.sum(1)

            if self.gat is not None:
                if self.node_att is not None:
                    rel_emb = self.relationEmbedding(attentive_v_emb, q_sum, bbox)
                else:
                    rel_emb = self.relationEmbedding(v_emb, q_sum, bbox)
                # GAT
                relational_v_emb, att = self.gat(q_sum, v_emb, rel_emb)
                attention['gat_att'] = att
                #v_emb = v_emb + relational_v_emb
            else:
                relational_v_emb = torch.zeros_like(v_emb)



            if self.spatial_fea_proj is not None:
                spatial = self.spatial_fea_proj(spatial).squeeze()# self.spatial_fea_proj(spatial) # [B, v_dim]])
                spatial = spatial.unsqueeze(1).repeat(1, n_obj, 1) #/ n_obj # [B, 1, v_dim]
            else:
                spatial = torch.zeros_like(v_emb)

            if self.gat is None and self.node_att is None:
                v_emb = v_emb
            else:
                if self.gat is not None and self.node_att is not None:
                    v_emb = relational_v_emb + attentive_v_emb
                else:
                    v_emb = relational_v_emb + attentive_v_emb

            if self.spatial_fea_proj is not None:
                v_emb = self.gate2(v_emb, spatial)
            else:
                v_emb = v_emb + spatial

            step = step + 1


        return v_emb, q_emb, attention

class SpatialFeatureProj(nn.Module):
    def __init__(self, v_dim):
        super(SpatialFeatureProj, self).__init__()
        self.conv1 = nn.Conv2d(v_dim, v_dim, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(v_dim)
        self.pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.pool(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        return out

class RN(nn.Module):
    def __init__(self, v_dim, q_dim, subspace_dim, r_dim, pe_enable=True, dropout=.2, wn=True):
        super(RN, self).__init__()
        self.pe_enable = pe_enable
        self.r_dim = r_dim

        conv_channels = subspace_dim
        if pe_enable:
            v_dim = v_dim + 4
        self.v_prj = FCNet([v_dim, conv_channels], dropout=dropout, wn=wn)
        self.q_prj = FCNet([q_dim, conv_channels], dropout=dropout, wn=wn)
        out_channel1 = r_dim
        out_channel2 = r_dim
        self.r_conv01 = nn.Conv2d(in_channels=conv_channels, out_channels=out_channel1, kernel_size=1)
        self.r_conv02 = nn.Conv2d(in_channels=out_channel1, out_channels=out_channel2, kernel_size=1)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        if not wn:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()
                if isinstance(m, nn.Conv2d):
                    init.kaiming_uniform_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()


    def forward(self, X, Q, pos=None):
        '''
        :param X: [batch_size, vloc, in_dim]
        :param Q: [bs, qdim]
        :param pos: position [bs, vloc, 4] x,y,w,h
        :return: r_emb: [bs, v_loc, v_loc, r_dim]
        '''
        if self.pe_enable:
            X = torch.cat([X, pos], dim=2)
        bs, vloc, in_dim = X.size()

        self.Nr = vloc

        # project the visual features and get the relation map
        X = self.v_prj(X)#[bs, Nr, subspace_dim]
        Q = self.q_prj(Q).unsqueeze(1)#[bs, 1, subspace_dim]
        X = X + Q
        Xi = X.unsqueeze(1).repeat(1,self.Nr,1,1)#[bs, Nr, Nr, subspace_dim]
        Xj = X.unsqueeze(2).repeat(1,1,self.Nr,1)#[bs, Nr, Nr, subspace_dim]
        X = Xi * Xj #[bs, Nr, Nr, subspace_dim]
        X = X.permute(0, 3, 1, 2)#[bs, subspace_dim, Nr, Nr]

        X = self.drop(self.relu(self.r_conv01(X)))
        r_emb = self.drop(self.relu(self.r_conv02(X))) # [bs, r_dim, Nr, Nr]
        r_emb = r_emb.permute(0, 2, 3, 1) # [bs, Nr, Nr, r_dim]
        return r_emb


class SelectGate(nn.Module):
    def __init__(self, v_dim, dropout=0.2, wn=True):
        super(SelectGate, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = FCNet([2*v_dim, 2*v_dim], relu=False, wn=wn)
        self.fc2 = FCNet([2*v_dim, v_dim], relu=False, wn=wn)

    def forward(self, v1, v2):
        '''
        :param v1: [B, N, v_dim]
        :param v2: [B, N, v_dim]
        :return: v: [B, N, v_dim]
        '''
        v = self.dropout(torch.cat([v1, v2], dim=-1))#[B, N, 2*v_dim]
        gate = torch.sigmoid(self.fc1(v))#[B, N, 2*v_dim]
        v = gate * v
        v = self.fc2(self.dropout(v))#[B, N, v_dim]
        return v