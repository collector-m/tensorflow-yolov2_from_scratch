#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : dataset.py
#   Author      : YunYang1994
#   Created date: 2019-01-12 21:07:25
#   Description :
#
#================================================================

import numpy as np
import tensorflow as tf
from core import utils

class parser(object):

    def __init__(self, IMAGE_H, IMAGE_W, GRID_H, GRID_W, ANCHORS, NUM_CLASSES,
                 TRUE_BOX_BUFFER=20, DEBUG=False):
        self.IMAGE_H = IMAGE_H
        self.IMAGE_W = IMAGE_W
        self.GRID_H  = GRID_H
        self.GRID_W  = GRID_W
        self.ANCHORS = ANCHORS
        self.NUM_CLASSES = NUM_CLASSES
        self.TRUE_BOX_BUFFER = TRUE_BOX_BUFFER
        self.DEBUG = DEBUG

    def preprocess(self, image, gt_boxes):
        image, gt_boxes = utils.resize_image_correct_bbox(image, gt_boxes, self.IMAGE_H, self.IMAGE_W)
        if self.DEBUG: return image, gt_boxes
        # data augmentation (continue to work)

        image = image / 255.
        y_true, true_boxes = tf.py_func(self.preprocess_true_boxes,
                                        inp=[gt_boxes], Tout=[tf.float32, tf.float32])
        return image, y_true, true_boxes

    def parser_example(self, serialized_example):

        features = tf.parse_single_example(
            serialized_example,
            features = {
                'image' : tf.FixedLenFeature([], dtype = tf.string),
                'boxes': tf.FixedLenFeature([], dtype = tf.string),
            }
        )

        image = tf.image.decode_jpeg(features['image'], channels = 3)
        image = tf.image.convert_image_dtype(image, tf.uint8)

        gt_boxes = tf.decode_raw(features['boxes'], tf.float32)
        gt_boxes = tf.reshape(gt_boxes, shape=[-1,5])

        return self.preprocess(image, gt_boxes)


    def preprocess_true_boxes(self, gt_boxes):

        NUM_ANCHORS = len(self.ANCHORS) // 2
        b_batch = np.zeros(shape=[1, 1, 1, self.TRUE_BOX_BUFFER, 4], dtype = np.float32)
        y_batch = np.zeros(shape=[self.GRID_H, self.GRID_W, NUM_ANCHORS, 5+self.NUM_CLASSES], dtype=np.float32)

        anchors = np.ones(shape=[NUM_ANCHORS, 4])
        anchors[:, 2:4] = np.array([self.ANCHORS]).reshape(-1, 2)

        for i in range(len(gt_boxes)):
            box = gt_boxes[i]
            # print(box)
            center_x = .5 * (box[0] + box[2])
            center_x = center_x / (float(self.IMAGE_W) / self.GRID_W) # uint: grid cell
            center_y = .5 * (box[1] + box[3])
            center_y = center_y / (float(self.IMAGE_H) / self.GRID_H) # uint: grid cell

            grid_x = int(np.floor(center_x))
            grid_y = int(np.floor(center_y))

            if grid_x < self.GRID_W and grid_y < self.GRID_H:
                cls_idx = int(box[4])
                center_w = (box[2] - box[0]) / (float(self.IMAGE_W) / self.GRID_W) # unit: grid cell
                center_h = (box[3] - box[1]) / (float(self.IMAGE_H) / self.GRID_H) # unit: grid cell

                box = np.array([center_x, center_y, center_w, center_h])
                anchors[:, 0:2] = np.array([center_x, center_y])

                # Find best anchor for each true box
                intersect_mins = np.maximum(box[0:2] - .5*box[2:4], anchors[:, 0:2] - .5*anchors[:, 2:4])
                intersect_maxs = np.minimum(box[0:2] + .5*box[2:4], anchors[:, 0:2] + .5*anchors[:, 2:4])
                intersect_wh   = np.maximum(intersect_maxs - intersect_mins, 0.)
                intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

                box_area = box[2] * box[3]
                anchors_area = anchors[:, 2] * anchors[:, 3]
                iou = intersect_area / (box_area + anchors_area - intersect_area + 1e-6)
                best_anchor = np.argmax(iou)
                y_batch[grid_y, grid_x, best_anchor, 0:4      ] = box
                y_batch[grid_y, grid_x, best_anchor, 4        ] = 1.
                y_batch[grid_y, grid_x, best_anchor, 5+cls_idx] = 1.

                b_batch[0, 0, 0, i] = box

        return y_batch, b_batch

class dataset(object):
    def __init__(self, parser, tfrecords_path, batch_size, shuffle=None):
        self.parser = parser
        self.filenames = tf.gfile.Glob(tfrecords_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._buildup()

    def _buildup(self):
        try:
            self._TFRecordDataset = tf.data.TFRecordDataset(self.filenames)
        except:
            raise NotImplementedError("No tfrecords found!")

        self._TFRecordDataset = self._TFRecordDataset.map(map_func = self.parser.parser_example,
                                                        num_parallel_calls = 10).repeat()
        if self.shuffle is not None:
            self._TFRecordDataset = self._TFRecordDataset.shuffle(self.shuffle)
        self._TFRecordDataset = self._TFRecordDataset.batch(self.batch_size).prefetch(self.batch_size)
        self._iterator = self._TFRecordDataset.make_one_shot_iterator()

    def get_next(self):
        return self._iterator.get_next()


