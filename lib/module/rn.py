import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from lib.module.fc import FCNet
import numpy as np

class RN(nn.Module):
    def __init__(self, v_dim, q_dim, subspace_dim, relation_glimpse, ksize=3, pe_enable=True, dropout_ratio=.2):
        super(RN, self).__init__()
        self.pe_enable = pe_enable
        self.relation_glimpse = relation_glimpse

        conv_channels = subspace_dim

        if pe_enable:
            v_dim = v_dim + 4
        self.v_prj = FCNet([v_dim, conv_channels], dropout=dropout_ratio)
        self.q_prj = FCNet([q_dim, conv_channels], dropout=dropout_ratio)
        out_channel1 = int(conv_channels/2)
        out_channel2 = int(conv_channels/4)
        if ksize == 3:
            padding1, padding2, padding3 = 1, 2, 4
        if ksize == 5:
            padding1, padding2, padding3 = 2, 4, 8
        if ksize == 7:
            padding1, padding2, padding3 = 3, 6, 12
     #   self.r_conv01 = nn.Conv2d(in_channels=conv_channels, out_channels=out_channel1, kernel_size=1)
     #   self.r_conv02 = nn.Conv2d(in_channels=out_channel1, out_channels=out_channel2, kernel_size=1)
    #    self.r_conv03 = nn.Conv2d(in_channels=out_channel2, out_channels=relation_glimpse, kernel_size=1)
        self.r_conv1 = (nn.Conv2d(in_channels=conv_channels, out_channels=out_channel1, kernel_size=ksize, dilation=1, padding=padding1))
        self.r_conv2 = (nn.Conv2d(in_channels=out_channel1, out_channels=out_channel2, kernel_size=ksize, dilation=2, padding=padding2))
        self.r_conv3 = (nn.Conv2d(in_channels=out_channel2, out_channels=relation_glimpse, kernel_size=ksize, dilation=4, padding=padding3))
        self.drop = nn.Dropout(dropout_ratio)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, X, Q, pos):
        '''
        :param X: [batch_size, vloc, in_dim]
        :param Q: [bs, qdim]
        :param pos: position [bs, vloc, 6] x,y,?,?,w,h
        :return: relation map:[batch_size, relation_glimpse, Nr, Nr]
                 relational_x: [bs, Nr, in_dim]
        '''
        X_ = X.clone()
        if self.pe_enable:
            X = torch.cat([X, pos[:,:,[0,1,4,5]]], dim=2)
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

        X0 = self.drop(self.relu(self.r_conv01(X)))
        X0 = self.drop(self.relu(self.r_conv02(X0)))
        relation_map0 = self.drop(self.relu(self.r_conv03(X0)))
        relation_map0 = relation_map0 + relation_map0.transpose(2, 3)
        relation_map0 = nn.functional.softmax(relation_map0.view(bs, self.relation_glimpse, -1), 2)
        relation_map0 = relation_map0.view(bs, self.relation_glimpse, self.Nr, -1)

        X = self.drop(self.relu(self.r_conv1(X)))#[bs, subspace_dim, Nr, Nr]
        X = self.drop(self.relu(self.r_conv2(X)))  # [bs, subspace_dim, Nr, Nr]
        relation_map = self.drop(self.relu(self.r_conv3(X)))  # [bs, relation_glimpse, Nr, Nr]
        relation_map = relation_map + relation_map.transpose(2, 3)
        relation_map = nn.functional.softmax(relation_map.view(bs, self.relation_glimpse, -1), 2)
        relation_map = relation_map.view(bs, self.relation_glimpse, self.Nr, -1)

        relational_X = torch.zeros_like(X_)
        for g in range(self.relation_glimpse):
            relational_X = relational_X + torch.matmul(relation_map[:,g,:,:], X_) + torch.matmul(relation_map0[:,g,:,:], X_)
        relational_X = relational_X/(2*self.relation_glimpse) #(relational_X/self.relation_glimpse + self.nonlinear(X_))/2
        return relation_map, relational_X

if __name__ == '__main__':
    vloc = 13
    qlen = 4
    bs = 8
    indim = 6
    x = Variable(torch.randn(bs, vloc, indim))
    att = Variable(torch.randn(bs, vloc, qlen))
    pos = Variable(torch.randn(bs, vloc, 6))

    rn = RN(in_dim=indim, subspace_dim=3, relation_glimpse=1, Nr=5)
    xx = rn(x, att, pos)




