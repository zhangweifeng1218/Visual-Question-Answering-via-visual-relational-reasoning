## Introduction
This repository is the implementation of ["Reasoning on the Relation: Enhancing Visual Representation for Visual Question Answering and Cross-Modal Retrieval"](https://ieeexplore.ieee.org/document/8988148)(IEEE Transactions on Multimedia, 2020)

![Model overview](https://github.com/zhangweifeng1218/Visual-Question-Answering-via-visual-relational-reasoning/raw/blob/master/model.png)

## Prerequisites
You may need a machine with GPUs (>=20GB memory), with PyTorch v1.0 for Python 3
we also use the block lib which can be installed as :

    pip install block.bootstrap.pytorch

## Preprocessing
First, you need download the [VQA 2.0 dataset](https://visualqa.org/download.html) and put them under /data/vqa directory

Our implementation uses the pretrained features from [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention), the fixed 36 features per image. After download these files, unzip these files and put them under /data/vqa/object directory. 

In addition, the glove.6B.zip is also need which can be downloaded [here](https://nlp.stanford.edu/projects/glove/). Also unzip the downloaded file and put them under /data/glove directory.

We use a part of Visual Genome dataset for data augmentation. The [image meta data](https://visualgenome.org/static/data/dataset/image_data.json.zip) and the [question answers of Version 1.2](https://visualgenome.org/static/data/dataset/question_answers.json.zip) are needed to be placed in data/vqa.

Then you need to run the following script to preprocessing the dataset:

    python script/vqa/preprocess.py

## Training
    python train_rn.py
the training hyper-parameters can be modified in config_rn.py

## Testing
    python infer_rn.py
 this will generate a json file containing the predicted answers for all testing sample.
 
 
 ## Citation
 If this repository is helpful for your research, we'd really appreciate it if you could cite the following paper:
 
    @article{zhang_vqa_rn,
       author={Jing, Yu, and Weifeng, Zhang, and Yuhang, Lu, and Zengchang, Qin, and Yue, Hu, and Jianlong, Tan, and Qi, Wu},
       title={Reasoning on the Relation: Enhancing Visual Representation for Visual Question Answering and Cross-Modal Retrieval},
       journal={IEEE Transactions on Multimedia},
       volume={22},
       number={12},
       pages={3196 - 3209},
       year={2020}}
