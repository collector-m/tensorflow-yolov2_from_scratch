#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-01-11 13:55:03
#   Description :
#
#================================================================

import numpy as np
import tensorflow as tf
from core import backbone, utils

WARM_UP_BATCHES  = 0
BATCH_SIZE       = 1
GRID_H , GRID_W  = 13, 13
IMAGE_H, IMAGE_W = 416, 416
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
NUM_ANCHORS      = len(ANCHORS) // 2
CLASSES          = [ cls.strip('\n') for cls in open("./data/voc.names").readlines()]
NUM_CLASSES      = len(CLASSES)
CLASS_WEIGHTS    = np.ones(NUM_CLASSES, dtype='float32')
TRUE_BOX_BUFFER  = 50

tfrecord = "../voc/voc.tfrecords"
sess = tf.Session()

paser = utils.parser(IMAGE_H, IMAGE_W, GRID_H, GRID_W, ANCHORS, NUM_CLASSES, DEBUG=False)
dataset = tf.data.TFRecordDataset(filenames = tf.gfile.Glob(tfrecord))
dataset = dataset.map(paser.parser_example, num_parallel_calls = 10)
dataset = dataset.repeat().shuffle(1).batch(BATCH_SIZE).prefetch(BATCH_SIZE)
iterator = dataset.make_one_shot_iterator()
example = iterator.get_next()
input_image, y_true, true_boxes = example

feature_extractor = backbone.FullYoloFeature(input_image)
features = feature_extractor.feature

output = tf.keras.layers.Conv2D(NUM_ANCHORS * (5 + NUM_CLASSES),
                (1,1), strides=(1,1),
                padding='same',
                name='detection_layer',
                kernel_initializer='lecun_normal')(features)
y_pred = tf.reshape(output, shape=[BATCH_SIZE, GRID_H, GRID_W, NUM_ANCHORS, 5+NUM_CLASSES])

loss_items = utils.compute_loss(y_true, y_pred, true_boxes, GRID_H, GRID_W, BATCH_SIZE, ANCHORS, CLASS_WEIGHTS)
sess.run(tf.global_variables_initializer())

# saver = tf.train.Saver(max_to_keep=2)
# write_op = tf.summary.merge_all()
# writer_train = tf.summary.FileWriter("./data/log")

# tf.summary.scalar("loss/loss_xy",      loss_items[1])
# tf.summary.scalar("loss/loss_wh",      loss_items[2])
# tf.summary.scalar("loss/loss_conf",    loss_items[3])
# tf.summary.scalar("loss/loss_class",   loss_items[4])
# tf.summary.scalar("yolov2/total_loss", loss_items[0])
# tf.summary.scalar("yolov2/recall_50",  loss_items[5])
# tf.summary.scalar("yolov2/recall_75",  loss_items[6])






