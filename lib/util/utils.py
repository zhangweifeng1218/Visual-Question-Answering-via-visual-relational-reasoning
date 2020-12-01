# File: utils.py
# Author: Ronil Pancholia
# Date: 4/22/19
# Time: 7:57 PM
import pickle
import sys
import os
import numpy as np

def count_parameters(model):
    return sum(p.nelement() for p in model.parameters() if p.requires_grad)

embedding_weights = None
def get_or_load_embeddings(word_dic_file, embedding_file):
    global embedding_weights, id2word

    if embedding_weights is not None:
        return embedding_weights

    with open(word_dic_file, 'rb') as f:
        dic = pickle.load(f)

    word2id = dic['word_dic']

    embed_size = 300
    vocab_size = len(word2id)
    sd = 1 / np.sqrt(embed_size)
    embedding_weights = np.random.normal(0, scale=sd, size=[vocab_size, embed_size])
    embedding_weights = embedding_weights.astype(np.float32)
    # "data/glove/glove.6B.300d.txt"
    with open(embedding_file, encoding="utf-8", mode="r") as textFile:
        for line in textFile:
            line = line.split()
            word = line[0]

            id = word2id.get(word, None)
            if id is not None:
                embedding_weights[id] = np.array(line[1:], dtype=np.float32)

    return embedding_weights

'''
id2word = []
embedding_weights = None

def get_or_load_embeddings(dataset_type='gqa', balanced=True):
    global embedding_weights, id2word

    if embedding_weights is not None:
        return embedding_weights

    with open('data/{}_dic.pkl'.format(dataset_type), 'rb') as f:
        dic = pickle.load(f)

    ans_dic_name = 'answer_dic_{}'.format('balanced' if balanced else 'all')
    id2word = set(dic['word_dic'].keys())
    id2word.update(set(dic[ans_dic_name].keys()))

    word2id = {word: id for id, word in enumerate(id2word)}

    embed_size = 300
    vocab_size = len(id2word)
    sd = 1 / np.sqrt(embed_size)
    embedding_weights = np.random.normal(0, scale=sd, size=[vocab_size, embed_size])
    embedding_weights = embedding_weights.astype(np.float32)

    with open("data/glove/glove.6B.300d.txt", encoding="utf-8", mode="r") as textFile:
        for line in textFile:
            line = line.split()
            word = line[0]

            id = word2id.get(word, None)
            if id is not None:
                embedding_weights[id] = np.array(line[1:], dtype=np.float32)

    return embedding_weights
'''

class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.items():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)