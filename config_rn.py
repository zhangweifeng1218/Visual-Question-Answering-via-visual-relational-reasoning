RN_vqa1 = {
    'train_split': 'train',
    'val_split': 'val',
    'test_split': 'test2015',
    'test_check_point': 'checkpoint/RN_vqa1/checkpoint_20.pth',
    'n_obj': 36,
    'n_word': 14,

    'wn': True,

    'n_vocab': 0,
    'classes': 0,
    'v_dim': 2048,

    # language
    'word_embedding_dim': 300,
    'rnn_type': 'GRU',
    'rnn_dim': 1024,
    'rnn_layer': 1,
    'rnn_bidirection': False,

    'dropout': 0.2,

    # rn
    'rn_sub_dim': 256,
    'pe_enable': True,
    'ksize': 3,

    # att
    'att_enable': True,

    # fuseion
    'fused_dim': 1024,

    'classifier_hid_dim': 2048,

    'dataset_type': 'vqa',
    'dictionary_file': 'data/vqa_dic.pkl',
    'word_dic_file': 'data/vqa/word2id.pkl',
    'embedding_file': 'data/glove/glove.6B.300d.txt',
    'batch_size': 512,
    'n_epoch': 20,
    'init_lr': 3e-3,
    'use_vg': True,
    'use_both': True,
    'train_check_point': None,
    'model_name': 'RN',
    'preload': False
}

RN_vqa2 = {
    'train_split': 'train',
    'val_split': 'val',
    'test_split': 'test2015',
    'test_check_point': 'checkpoint/RN_vqa1/checkpoint_20.pth',
    'n_obj': 36,
    'n_word': 14,

    'wn': True,

    'n_vocab': 0,
    'classes': 0,
    'v_dim': 2048,

    # language
    'word_embedding_dim': 300,
    'rnn_type': 'GRU',
    'rnn_dim': 1024,
    'rnn_layer': 1,
    'rnn_bidirection': False,

    'dropout': 0.2,

    # rn
    'rn_sub_dim': 256,
    'pe_enable': True,
    'ksize': 5,

    # att
    'att_enable': False,

    # fuseion
    'fused_dim': 1024,

    'classifier_hid_dim': 2048,

    'dataset_type': 'vqa',
    'dictionary_file': 'data/vqa_dic.pkl',
    'word_dic_file': 'data/vqa/word2id.pkl',
    'embedding_file': 'data/glove/glove.6B.300d.txt',
    'batch_size': 512,
    'n_epoch': 20,
    'init_lr': 3e-3,
    'use_vg': False,
    'use_both': False,
    'train_check_point': None,
    'model_name': 'RN',
    'preload': False
}
RN_vqa3 = {
    'train_split': 'train',
    'val_split': 'val',
    'test_split': 'test2015',
    'test_check_point': 'checkpoint/RN_vqa1/checkpoint_20.pth',
    'n_obj': 36,
    'n_word': 14,

    'wn': True,

    'n_vocab': 0,
    'classes': 0,
    'v_dim': 2048,

    # language
    'word_embedding_dim': 300,
    'rnn_type': 'GRU',
    'rnn_dim': 1024,
    'rnn_layer': 1,
    'rnn_bidirection': False,

    'dropout': 0.2,

    # rn
    'rn_sub_dim': 256,
    'pe_enable': True,
    'ksize': 7,

    # att
    'att_enable': False,

    # fuseion
    'fused_dim': 1024,

    'classifier_hid_dim': 2048,

    'dataset_type': 'vqa',
    'dictionary_file': 'data/vqa_dic.pkl',
    'word_dic_file': 'data/vqa/word2id.pkl',
    'embedding_file': 'data/glove/glove.6B.300d.txt',
    'batch_size': 512,
    'n_epoch': 20,
    'init_lr': 3e-3,
    'use_vg': False,
    'use_both': False,
    'train_check_point': None,
    'model_name': 'RN',
    'preload': False
}
RN_vqa4 = {
    'train_split': 'train',
    'val_split': 'val',
    'test_split': 'test2015',
    'test_check_point': 'checkpoint/RN_vqa4/checkpoint_20.pth',
    'n_obj': 36,
    'n_word': 14,

    'wn': True,

    'n_vocab': 0,
    'classes': 0,
    'v_dim': 2048,

    # language
    'word_embedding_dim': 300,
    'rnn_type': 'GRU',
    'rnn_dim': 1024,
    'rnn_layer': 1,
    'rnn_bidirection': False,

    'dropout': 0.2,

    # rn
    'rn_sub_dim': 256,
    'pe_enable': True,
    'ksize': 3,

    # att
    'att_enable': True,

    # fuseion
    'fused_dim': 1024,

    'classifier_hid_dim': 2048,

    'dataset_type': 'vqa',
    'dictionary_file': 'data/vqa_dic.pkl',
    'word_dic_file': 'data/vqa/word2id.pkl',
    'embedding_file': 'data/glove/glove.6B.300d.txt',
    'batch_size': 512,
    'n_epoch': 1,
    'init_lr': 3e-3,
    'use_vg': True,
    'use_both': True,
    'train_check_point': None,
    'model_name': 'RN',
    'preload': False
}

cfg_dict = {
    'RN_vqa1': RN_vqa1,  # full model 
    'RN_vqa2': RN_vqa2,  # only rn, 5x5 conv 
    'RN_vqa3': RN_vqa3,  # only rn, 7x7      
    'RN_vqa4': RN_vqa4, # use mfh
}