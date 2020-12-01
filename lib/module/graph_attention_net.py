import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.module.fc import FCNet


class GraphAttNet(nn.Module):

    def __init__(self, q_dim, ent_dim, rel_dim, att_hid_dim, out_dim,
                 n_att=1, multi_head_type="concat", dropout=0.2, wn=True):
        super(GraphAttNet, self).__init__()

        self.out_dim = out_dim
        self.self_att = GATSelfAtt(n_att, ent_dim+rel_dim, q_dim, att_hid_dim, dropout, wn)
        self.n_att = n_att
        self.multi_head_type = multi_head_type
        if self.multi_head_type == "concat":
            assert out_dim % n_att == 0
            transform = [ FCNet([ent_dim+rel_dim, int(out_dim/n_att)], wn=wn) for _ in range(n_att) ]
        else:
            transform = [ FCNet([ent_dim+rel_dim, out_dim], wn=wn) for _ in range(n_att) ]
        self.transform = nn.ModuleList(transform)

    def forward(self, q_emb, ent_emb, rel_emb):
        """
        :param q_emb: [ B, q_dim ]
        :param ent_emb: [ B, n_ent, ent_dim ]
        :param rel_emb: [ B, n_sbj, n_obj, rel_dim ]
        return updated ent_emb: [B, n_sbj, out_dim]
        """
        B, n_sbj, n_obj, rel_dim = rel_emb.size()
        ent_exp = ent_emb.unsqueeze(2).expand(-1, -1, n_obj, -1)
        joint = torch.cat([ent_exp, rel_emb], dim=-1)
        # [B, n_sbj, n_obj, ent_dim+rel_dim]

        updated = [self.transform[i](joint) for i in range(self.n_att)] # list: [B, n_sbj, n_obj, out_dim/n_att] x n_att
        updated = torch.stack(updated, dim=-2)  # [B, n_sbj, n_obj, n_att, out_dim/n_att]

        att = self.self_att(joint, q_emb) #  [B, n_sbj, n_obj, n_att]
        updated = updated * att.unsqueeze(-1)

        if self.multi_head_type == "concat":
            updated = updated.view(B, n_sbj, n_obj, self.out_dim) # [B, n_sbj, n_obj, out_dim]
        else:
            updated = updated.sum(dim=-2)
        updated = updated.sum(dim=2) # [B, n_sbj, out_dim]
        return updated, att

class GATSelfAtt(nn.Module):
    '''self attention in GAT'''
    def __init__(self, n_att, ent_dim, q_dim, hid_dim, dropout=0.5, wn=True):
        super(GATSelfAtt, self).__init__()
        self.n_att = n_att
        self.h_dim = hid_dim
        self.transform_W = FCNet([ent_dim, self.h_dim], bias=False, relu=False, wn=wn)
        self.transform_A = FCNet([self.h_dim, n_att], bias=False, relu=False, wn=wn)
        self.transform_Q = FCNet([q_dim, self.h_dim], bias=False, relu=False, wn=wn)
        self.leakyReLU = nn.LeakyReLU()

    def forward(self, ent_emb, q_emb):
        '''
        :param ent_emb: [B, n_sbj, n_obj, ent_dim]
        :param q_emb: [B, q_dim]
        :return: attention:  [B, n_sbj, n_obj, n_att]
        '''
        attentions = self.leakyReLU(self.transform_A(self.transform_Q(q_emb).unsqueeze(1).unsqueeze(2) * self.transform_W(ent_emb) * self.transform_W(ent_emb.transpose(1, 2))))
        # attentions:[B, n_sbj, n_obj, n_att]
        attentions = F.softmax(attentions, dim=2)
        return attentions

class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(1, 1, n_state))
        self.b = nn.Parameter(torch.zeros(1, 1, n_state))
        self.e = e

    def forward(self, x):
        # x: [B, n_sbj, dim]
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e) # [B, n_sbj, dim]
        return self.g * x + self.b

