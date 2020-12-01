"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function
import os
import json
import _pickle as cPickle
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
import torch
from torch.utils.data import Dataset
import script.vqa.compute_softscore
import itertools
import lib.data.vqa.utils as utils

COUNTING_ONLY = False


# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
            ('number of' in q.lower() and 'number of the' not in q.lower()) or \
            'amount of' in q.lower() or \
            'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK for Visual Genome dataset
                tokens.append(self.word2idx.get(w, self.padding_idx - 1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    if None != answer:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'answer': answer}
    return entry


def _load_dataset(dataroot, name, img_id2val, label2ans, vqa_version=2):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test-dev2015', test2015'
    """
    question_path = os.path.join(
        dataroot, 'questions', 'v%d_OpenEnded_mscoco_%s_questions.json' % \
                  (vqa_version, name + '2014' if 'test' != name[:4] else name))
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    if 'test' != name[:4]:  # train, val
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
        answers = cPickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])

        utils.assert_eq(len(questions), len(answers))
        entries = []
        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            if not COUNTING_ONLY or is_howmany(question['question'], answer, label2ans):
                entries.append(_create_entry(img_id2val[img_id], question, answer))
    else:  # test2015
        entries = []
        for question in questions:
            img_id = question['image_id']
            if not COUNTING_ONLY or is_howmany(question['question'], None, None):
                entries.append(_create_entry(img_id2val[img_id], question, None))

    return entries


def _load_visualgenome(dataroot, name, img_id2val, label2ans, adaptive=False):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
   # train_ids_file =  'data/vqa/object/{}_ids.pkl'.format(name)
   # imgids = cPickle.load(open(train_ids_file, 'rb'))
    question_path = os.path.join(dataroot, 'questions', 'VG_questions.json')
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])

    answer_path = os.path.join(dataroot, 'cache', 'vg_target.pkl')
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']
        if img_id in img_id2val.keys():
            if not COUNTING_ONLY or is_howmany(question['question'], answer, label2ans):
                entries.append(_create_entry(img_id2val[img_id], question, answer))

    return entries


def _find_coco_id(vgv, vgv_id):
    for v in vgv:
        if v['id'] == vgv_id:
            return v['coco_id']
    return None


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, dataroot='data/vqa', adaptive=False):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'test-dev2015', 'test2015']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.adaptive = adaptive

        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, 'object', '%s_imgid2idx.pkl' % (name)), 'rb'))

        h5_path = os.path.join(dataroot, 'object', '%s.hdf5' % (name))

        print('loading features from h5 file')
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))

        with open(os.path.join(dataroot, 'spatial', '{}_spatial_info.json'.format('test' if name == 'test2015' else name)), 'r') as f:
            self.global_fea_id2idx = json.load(f)

        with h5py.File(os.path.join(dataroot, 'spatial', '{}_spatial.hdf5'.format('test' if name == 'test2015' else name)), 'r') as hf:
            self.global_fea = np.array(hf.get('data'))

        self.entries = _load_dataset(dataroot, name, self.img_id2idx, self.label2ans, vqa_version=2)
        self.tokenize()
        self.tensorize()
        self.v_dim = self.features.size(1 if self.adaptive else 2)
        self.s_dim = self.spatials.size(1 if self.adaptive else 2)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)
        self.global_fea = torch.from_numpy(self.global_fea)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            if None != answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        if not self.adaptive:
            features = self.features[entry['image']]
            spatials = self.spatials[entry['image']]

        image_id = str(entry['image_id'])
        global_fea = self.global_fea[self.global_fea_id2idx[image_id]]

        question = entry['q_token']
        question_id = entry['question_id']
        answer = entry['answer']
        if None != answer:
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            return global_fea, features, spatials, question, torch.LongTensor([question.size(0)]), target
        else:
            return global_fea, features, spatials, question, torch.LongTensor([question.size(0)]), question_id

    def __len__(self):
        return len(self.entries)


class VisualGenomeFeatureDataset(Dataset):
    def __init__(self, name, global_fea, global_fea_id2idx, features, spatials, dictionary, dataroot='data/vqa', adaptive=False, pos_boxes=None):
        super(VisualGenomeFeatureDataset, self).__init__()

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.adaptive = adaptive

        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, 'object', '{}_imgid2idx.pkl'.format(name)), 'rb'))

        self.global_fea = global_fea
        self.global_fea_id2idx = global_fea_id2idx
        self.features = features
        self.spatials = spatials
        if self.adaptive:
            self.pos_boxes = pos_boxes

        self.entries = _load_visualgenome(dataroot, name, self.img_id2idx, self.label2ans)
        self.tokenize()
        self.tensorize()
        self.v_dim = self.features.size(1 if self.adaptive else 2)
        self.s_dim = self.spatials.size(1 if self.adaptive else 2)

    def tokenize(self, max_length=14):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)
            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                entry['answer']['labels'] = labels
                entry['answer']['scores'] = scores
            else:
                entry['answer']['labels'] = None
                entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        if not self.adaptive:
            features = self.features[entry['image']]
            spatials = self.spatials[entry['image']]

        image_id = str(entry['image_id'])
        global_fea = self.global_fea[self.global_fea_id2idx[image_id]]

        question = entry['q_token']
        question_id = entry['question_id']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)
        return global_fea, features, spatials, question, torch.LongTensor([question.size(0)]), target

    def __len__(self):
        return len(self.entries)

# if __name__=='__main__':
#    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
#    tfidf, weights = tfidf_from_questions(['train', 'val', 'test2015'], dictionary)

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dictionary = Dictionary.load_from_file('data/vqa_dic.pkl')
    train_dset = VQAFeatureDataset('val', dictionary)
    # name = 'train'
    # eval_dset = VQAFeatureDataset(name, dictionary)
    # vg_dset = VisualGenomeFeatureDataset(name, eval_dset.features, eval_dset.spatials, dictionary)

    # train_loader = DataLoader(vg_dset, 10, shuffle=True, num_workers=1)

    loader = DataLoader(train_dset, 10, shuffle=True, num_workers=1)
    for i, (g, v, b, q, q_len, a) in enumerate(loader):
        print(v.size())
        print(q.size())
        print(a.size())

# VisualGenome Train
#     Used COCO images: 51487/108077 (0.4764)
#     Out-of-split COCO images: 17464/51487 (0.3392)
#     Used VG questions: 325311/726932 (0.4475)

# VisualGenome Val
#     Used COCO images: 51487/108077 (0.4764)
#     Out-of-split COCO images: 34023/51487 (0.6608)
#     Used VG questions: 166409/726932 (0.2289)
