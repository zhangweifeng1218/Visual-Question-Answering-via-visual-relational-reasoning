import multiprocessing
import pickle
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
import time
import _pickle as cPickle
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from lib.util.utils import count_parameters, Logger
from lib.data.vqa.dataset_vqa import VQAFeatureDataset, VisualGenomeFeatureDataset
import numpy as np
import random
from lib.model.model_dict import models
from config_rn import cfg_dict
from lib.data.vqa.dataset_vqa import Dictionary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def bce_with_logits(logits, labels):
    assert logits.dim() == 2
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1) # multiply by number of QAs
    return loss

def compute_score_with_logits(logits, labels):
    with torch.no_grad():
        logits = torch.max(logits, 1)[1] # argmax
        one_hots = torch.zeros(*labels.size()).cuda()
        one_hots.scatter_(1, logits.view(-1, 1), 1)
        scores = (one_hots * labels)
        return scores

def lr_schedule_func_builder(warmup=2, warmup_step=2, warmup_factor=0.2, keep_steps=7, decay_step=2, decay_ratio = 0.5):
    def func(epoch):
        alpha = float(warmup) / float(warmup_step)
        warmed_ratio = warmup_factor * (1. - alpha) + alpha
        if epoch <= warmup:
            alpha = float(epoch) / float(warmup_step)
            return warmup_factor * (1. - alpha) + alpha
        else:
            if epoch < warmup+keep_steps:
                return warmed_ratio
            else:
                idx = int((epoch-keep_steps)/decay_step)
                return pow(decay_ratio, idx) * warmed_ratio
    return func


def train(epoch):
    dataset = iter(train_loader)
    pbar = tqdm(dataset)
    epoch_score = 0.0
    epoch_loss = 0.0
    moving_score = 0
    moving_loss = 0
    n_samples = 0
    batch_cnt = 0
    net.train(True)
    tics = []
    t_start = time.time()
    for spatial, image, bbox, question, q_len, answer in pbar:
        spatial, image, bbox, question, q_len, answer = (
            spatial.to(device),
            image.to(device),
            bbox.to(device),
            question.to(device),
            q_len.to(device),
            answer.to(device),
        )
       # bbox = bbox[:, :, [0, 1, 4, 5]]
        q_len = q_len.squeeze()
        n_samples += image.size(0)

        net.zero_grad()
        output, _ = net(image, bbox, question, q_len)
        target = answer#torch.zeros_like(output).scatter_(1, answer.view(-1, 1), 1)
        loss = criterion(output, target)


        loss.backward()
        clip_grad_norm_(net.parameters(), 0.25)
        optimizer.step()

        batch_score = compute_score_with_logits(output, target).sum()
        epoch_loss += float(loss.data.item()) * target.size(0)
        epoch_score += float(batch_score)
        moving_loss = epoch_loss / n_samples
        moving_score = epoch_score / n_samples

        t_end = time.time()
        t = 1000*(t_end-t_start)
        tics.append(t)
        pbar.set_description(
            'Train Epoch: {}; Loss: {:.6f}; Acc: {:.6f}| {:.2f}ms'.format(epoch + 1, moving_loss, moving_score, t))
        t_start = time.time()

    logger.write('Epoch: {:2d}: Train Loss: {:.6f}; Train Acc: {:.4f} | {}({:.2f})'.format(epoch+1, moving_loss, moving_score, np.mean(tics), np.var(tics)))

def validate(epoch):
    dataset = iter(val_loader)
    pbar = tqdm(dataset)
    epoch_score = 0.0
    epoch_loss = 0.0
    moving_score = 0
    moving_loss = 0
    n_samples = 0
    net.eval()
    with torch.no_grad():
        for spatial, image, bbox, question, q_len, answer in pbar:
            spatial, image, bbox, question, q_len, answer = (
                spatial.to(device),
                image.to(device),
                bbox.to(device),
                question.to(device),
                q_len.to(device),
                answer.to(device),
            )
        #    bbox = bbox[:, :, [0, 1, 4, 5]]
            q_len = q_len.squeeze()
            n_samples += image.size(0)

            output, _ = net(image, bbox, question, q_len)
            target = answer  # torch.zeros_like(output).scatter_(1, answer.view(-1, 1), 1)
            loss = criterion(output, target)

            batch_score = compute_score_with_logits(output, target).sum()
            epoch_loss += float(loss.data.item()) * target.size(0)
            epoch_score += float(batch_score)
            moving_loss = epoch_loss / n_samples
            moving_score = epoch_score / n_samples

            pbar.set_description(
                'Val Epoch: {}; Loss: {:.6f}; Acc: {:.6f}'.format(epoch + 1, moving_loss, moving_score))


    logger.write('Val: {:2d}: Loss: {:.6f}; Acc: {:.4f}'.format(epoch+1, moving_loss, moving_score))


if __name__ == '__main__':
    cfg_name = 'RN_vqa1'  # str(sys.argv[1])
    cfg = cfg_dict[cfg_name]
    np.random.seed(101)  # numpy
    random.seed(101)  # random and transforms
    torch.manual_seed(101)
    torch.cuda.manual_seed(101)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file(cfg['dictionary_file'])
    with open(cfg['word_dic_file'], 'wb') as f:
        pickle.dump({'word_dic': dictionary.word2idx}, f)

    n_words = dictionary.ntoken
    cfg['n_vocab'] = n_words


    print('Loading data...')
    train_dataset = VQAFeatureDataset('train', dictionary)
    n_answers = train_dataset.num_ans_candidates
    cfg['classes'] = n_answers

    net = models[cfg['model_name']](cfg)
    n_params = count_parameters(net)
    print("model: {:,} M parameters".format(n_params / 1024 / 1024))

    val_dataset = VQAFeatureDataset('val', dictionary)

    if cfg['use_vg']:
        vg_train_dataset = VisualGenomeFeatureDataset('train', train_dataset.global_fea,
                                                      train_dataset.global_fea_id2idx,
                                                      train_dataset.features, train_dataset.spatials, dictionary)
        vg_val_dataset = VisualGenomeFeatureDataset('val', val_dataset.global_fea,
                                                      val_dataset.global_fea_id2idx,
                                                      val_dataset.features, val_dataset.spatials, dictionary)
        train_dataset = ConcatDataset([train_dataset, vg_train_dataset, vg_val_dataset])

    if cfg['use_both']:
        train_dataset = ConcatDataset([train_dataset, val_dataset])

    train_loader = DataLoader(train_dataset, cfg['batch_size'], shuffle=True, num_workers=1)

    if not cfg['use_both']:
        val_loader = DataLoader(val_dataset, cfg['batch_size'], shuffle=True, num_workers=1)



    print('Creating Model...')
    net = models[cfg['model_name']](cfg).to(device)
    net = nn.DataParallel(net)
    n_params = count_parameters(net)
    print("model: {:,} M parameters".format(n_params / 1024 / 1024))

    criterion = bce_with_logits#nn.CrossEntropyLoss()
    optimizer = optim.Adamax(net.parameters(), lr=cfg['init_lr'])
    sched = LambdaLR(optimizer, lr_lambda=lr_schedule_func_builder())

    checkpoint_path = 'checkpoint/{}'.format(cfg_name)
    if os.path.exists(checkpoint_path) is False:
        os.mkdir(checkpoint_path)

    if cfg['train_check_point'] is not None:
        net_checkpoint = torch.load(cfg['train_check_point'])
        net.load_state_dict(net_checkpoint)
        optim_checkpoint = torch.load('optim_{}'.format(cfg['train_check_point']))
        optimizer.load_state_dict(optim_checkpoint)

    logger = Logger(os.path.join(checkpoint_path, "log.txt"))
    for k, v in cfg.items():
        logger.write(k+': {}'.format(v))
    print('Starting train...')
    for epoch in range(cfg['n_epoch']):
        lr_strs = '; '.join(['group {} lr: {}'.format(i, param_group['lr']) for i, param_group in enumerate(optimizer.param_groups)])
        logger.write(lr_strs)
        train(epoch)
        if not cfg['use_both']:
            validate(epoch)
        sched.step(epoch)

        with open('{}/checkpoint_{}.pth'.format(checkpoint_path, str(epoch + 1).zfill(2)), 'wb') as f:
            torch.save(net.state_dict(), f)
        with open('{}/optim_checkpoint_{}.pth'.format(checkpoint_path, str(epoch + 1).zfill(2)), 'wb') as f:
            torch.save(optimizer.state_dict(), f)


