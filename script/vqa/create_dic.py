from __future__ import print_function
import os
import sys
import json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.data.vqa.dataset_vqa import Dictionary


def create_dictionary(dataroot, vqa_version=2):
    dictionary = Dictionary()
    questions = []
    files = [
        'v{}_OpenEnded_mscoco_train2014_questions.json'.format(vqa_version),
        'v{}_OpenEnded_mscoco_val2014_questions.json'.format(vqa_version),
        'v{}_OpenEnded_mscoco_test2015_questions.json'.format(vqa_version),
        'v{}_OpenEnded_mscoco_test-dev2015_questions.json'.format(vqa_version)
    ]
    for path in files:
        question_path = os.path.join(dataroot, path)
        qs = json.load(open(question_path))['questions']
        for q in qs:
            dictionary.tokenize(q['question'], True)
    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb

def create_dic():
    vqa_version = 2
    print('creating dictionary for VQA 2.0...')
    d = create_dictionary('data/vqa/questions', vqa_version)
    d.dump_to_file('data/vqa_dic.pkl')
    print('done: dictionary size {}'.format(len(d)))

if __name__ == '__main__':
    vqa_version = 2
    d = create_dictionary('data/vqa/questions', vqa_version)
    d.dump_to_file('data/vqa_dic.pkl')

    d = Dictionary.load_from_file('data/vqa_dic.pkl')
    emb_dim = 300
    glove_file = 'data/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save('data/glove6b_init_%dd.npy' % emb_dim, weights)
