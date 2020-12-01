"""
This code is from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa

Reads in a tsv file with pre-trained bottom up attention features and
stores it in HDF5 format.  Also store {image_id: feature_idx}
 as a pickle file.

Hierarchy of HDF5 file:

{ 'image_features': num_images x num_boxes x 2048 array of features
  'image_bb': num_images x num_boxes x 4 array of bounding boxes }
"""
from __future__ import print_function

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import base64
import csv
import h5py
import _pickle as cPickle
import numpy as np
import lib.data.vqa.utils as utils


csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features']
infile = 'data/vqa/object/trainval_36/trainval_resnet101_faster_rcnn_genome_36.tsv'
infile_test = 'data/vqa/object/test2015_36/test2015_resnet101_faster_rcnn_genome_36.tsv'
train_data_file = 'data/vqa/object/train.hdf5'
val_data_file = 'data/vqa/object/val.hdf5'
test_data_file = 'data/vqa/object/test2015.hdf5'
train_indices_file = 'data/vqa/object/train_imgid2idx.pkl'
val_indices_file = 'data/vqa/object/val_imgid2idx.pkl'
test_indices_file = 'data/vqa/object/test2015_imgid2idx.pkl'
train_ids_file = 'data/vqa/object/train_ids.pkl'
val_ids_file = 'data/vqa/object/val_ids.pkl'
test_ids_file = 'data/vqa/object/test2015_ids.pkl'
train_img_path = 'data/vqa/image/train'
val_img_path = 'data/vqa/image/val'
test_img_path = 'data/vqa/image/test'

feature_length = 2048
num_fixed_boxes = 36


def feature_convert():
    print('convert features...')
    h_train = h5py.File(train_data_file, "w")
    h_val = h5py.File(val_data_file, "w")
    h_test = h5py.File(test_data_file, "w")

    if os.path.exists(train_ids_file) and os.path.exists(val_ids_file) and os.path.exists(test_ids_file):
        train_imgids = cPickle.load(open(train_ids_file, 'rb'))
        val_imgids = cPickle.load(open(val_ids_file, 'rb'))
        test_imgids = cPickle.load(open(test_ids_file, 'rb'))
    else:
        train_imgids = utils.load_imageid(train_img_path)
        val_imgids = utils.load_imageid(val_img_path)
        test_imgids = utils.load_imageid(test_img_path)
        cPickle.dump(train_imgids, open(train_ids_file, 'wb'))
        cPickle.dump(val_imgids, open(val_ids_file, 'wb'))
        cPickle.dump(test_imgids, open(test_ids_file, 'wb'))

    train_indices = {}
    val_indices = {}
    test_indices = {}

    train_img_features = h_train.create_dataset(
        'image_features', (len(train_imgids), num_fixed_boxes, feature_length), 'f')
    train_img_bb = h_train.create_dataset(
        'image_bb', (len(train_imgids), num_fixed_boxes, 4), 'f')
    train_spatial_img_features = h_train.create_dataset(
        'spatial_features', (len(train_imgids), num_fixed_boxes, 6), 'f')

    val_img_bb = h_val.create_dataset(
        'image_bb', (len(val_imgids), num_fixed_boxes, 4), 'f')
    val_img_features = h_val.create_dataset(
        'image_features', (len(val_imgids), num_fixed_boxes, feature_length), 'f')
    val_spatial_img_features = h_val.create_dataset(
        'spatial_features', (len(val_imgids), num_fixed_boxes, 6), 'f')

    test_img_bb = h_test.create_dataset(
        'image_bb', (len(test_imgids), num_fixed_boxes, 4), 'f')
    test_img_features = h_test.create_dataset(
        'image_features', (len(test_imgids), num_fixed_boxes, feature_length), 'f')
    test_spatial_img_features = h_test.create_dataset(
        'spatial_features', (len(test_imgids), num_fixed_boxes, 6), 'f')

    train_counter = 0
    val_counter = 0
    test_counter = 0

    print("reading train/val tsv...")
    with open(infile, "r+") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in tqdm(reader):
            item['num_boxes'] = int(item['num_boxes'])
            item['boxes'] = bytes(item['boxes'], 'utf')
            item['features'] = bytes(item['features'], 'utf')
            image_id = int(item['image_id'])
            image_w = float(item['image_w'])
            image_h = float(item['image_h'])
            bboxes = np.frombuffer(
                base64.decodestring(item['boxes']),
                dtype=np.float32).reshape((item['num_boxes'], -1))

            box_width = bboxes[:, 2] - bboxes[:, 0]
            box_height = bboxes[:, 3] - bboxes[:, 1]
            scaled_width = box_width / image_w
            scaled_height = box_height / image_h
            scaled_x = bboxes[:, 0] / image_w
            scaled_y = bboxes[:, 1] / image_h

            box_width = box_width[..., np.newaxis]
            box_height = box_height[..., np.newaxis]
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]

            spatial_features = np.concatenate(
                (scaled_x,
                 scaled_y,
                 scaled_x + scaled_width,
                 scaled_y + scaled_height,
                 scaled_width,
                 scaled_height),
                axis=1)

            if image_id in train_imgids:
                train_imgids.remove(image_id)
                train_indices[image_id] = train_counter
                train_img_bb[train_counter, :, :] = bboxes
                train_img_features[train_counter, :, :] = np.frombuffer(
                    base64.decodestring(item['features']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                train_spatial_img_features[train_counter, :, :] = spatial_features
                train_counter += 1
            elif image_id in val_imgids:
                val_imgids.remove(image_id)
                val_indices[image_id] = val_counter
                val_img_bb[val_counter, :, :] = bboxes
                val_img_features[val_counter, :, :] = np.frombuffer(
                    base64.decodestring(item['features']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                val_spatial_img_features[val_counter, :, :] = spatial_features
                val_counter += 1
            else:
                assert False, 'Unknown image id: %d' % image_id

    print("reading test tsv...")
    with open(infile_test, "r+") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=FIELDNAMES)
        for item in tqdm(reader):
            item['num_boxes'] = int(item['num_boxes'])
            item['boxes'] = bytes(item['boxes'], 'utf')
            item['features'] = bytes(item['features'], 'utf')
            image_id = int(item['image_id'])
            image_w = float(item['image_w'])
            image_h = float(item['image_h'])
            bboxes = np.frombuffer(
                base64.decodestring(item['boxes']),
                dtype=np.float32).reshape((item['num_boxes'], -1))

            box_width = bboxes[:, 2] - bboxes[:, 0]
            box_height = bboxes[:, 3] - bboxes[:, 1]
            scaled_width = box_width / image_w
            scaled_height = box_height / image_h
            scaled_x = bboxes[:, 0] / image_w
            scaled_y = bboxes[:, 1] / image_h

            box_width = box_width[..., np.newaxis]
            box_height = box_height[..., np.newaxis]
            scaled_width = scaled_width[..., np.newaxis]
            scaled_height = scaled_height[..., np.newaxis]
            scaled_x = scaled_x[..., np.newaxis]
            scaled_y = scaled_y[..., np.newaxis]

            spatial_features = np.concatenate(
                (scaled_x,
                 scaled_y,
                 scaled_x + scaled_width,
                 scaled_y + scaled_height,
                 scaled_width,
                 scaled_height),
                axis=1)

            if image_id in test_imgids:
                test_imgids.remove(image_id)
                test_indices[image_id] = test_counter
                test_img_bb[test_counter, :, :] = bboxes
                test_img_features[test_counter, :, :] = np.frombuffer(
                    base64.decodestring(item['features']),
                    dtype=np.float32).reshape((item['num_boxes'], -1))
                test_spatial_img_features[test_counter, :, :] = spatial_features
                test_counter += 1
            else:
                assert False, 'Unknown image id: %d' % image_id

    if len(train_imgids) != 0:
        print('Warning: train_image_ids is not empty')

    if len(val_imgids) != 0:
        print('Warning: val_image_ids is not empty')

    if len(test_imgids) != 0:
        print('Warning: test_image_ids is not empty')

    cPickle.dump(train_indices, open(train_indices_file, 'wb'))
    cPickle.dump(val_indices, open(val_indices_file, 'wb'))
    cPickle.dump(test_indices, open(test_indices_file, 'wb'))
    h_train.close()
    h_val.close()
    h_test.close()
    print("done!")
