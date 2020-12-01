import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from lib.util.utils import get_or_load_embeddings

class WordEmbedding(nn.Module):
    """Word Embedding

    The n_tokens-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, n_tokens, emb_dim, dropout=0):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(n_tokens+1, emb_dim, padding_idx=n_tokens)
        self.dropout = nn.Dropout(dropout, inplace=True) if dropout > 0 else None
        self.n_tokens = n_tokens
        self.emb_dim = emb_dim

    def init_embedding(self, word_dic_file, embedding_file):
        weight_init = torch.Tensor(get_or_load_embeddings(word_dic_file, embedding_file))
        assert weight_init.shape == (self.n_tokens, self.emb_dim)
        self.emb.weight.data[:self.n_tokens] = weight_init

    def freeze(self):
        self.emb.weight.requires_grad = False

    def defreeze(self):
        self.emb.weight.requires_grad = True

    def forward(self, x):
        emb = self.emb(x)
        if self.dropout is not None: emb = self.dropout(emb)
        return emb

class QuestionEmbedding(nn.Module):
    """Module for question embedding
    """
    def __init__(self, in_dim, hid_dim, n_layers, rnn_type='GRU', keep_seq=False, bidirectional=False):

        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU

        self.rnn = rnn_cls(
            in_dim, hid_dim, n_layers,
            bidirectional=bidirectional,
            batch_first=True)

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.keep_seq = keep_seq  # if keep_seq is True, the output will be [B, seq_len, dim]
        self.ndirections = 1 + int(bidirectional)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.n_layers * self.ndirections, batch, self.hid_dim)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, emb, q_lens):
        # emb: [batch, sequence, in_dim]
        # q_lens: [batch, len]
        batch = emb.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        emb = nn.utils.rnn.pack_padded_sequence(emb, q_lens, batch_first=True)
        output, hidden = self.rnn(emb, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        if self.keep_seq:
            return output #[batch, sequence, hidden*ndirections]

        if self.ndirections == 1:
            return output[:, -1] #[batch, hidden]

        forward_ = output[:, -1, :self.hid_dim]
        backward = output[:, 0, self.hid_dim:]
        return torch.cat((forward_, backward), dim=1) #[batch, hidden*ndirections]
