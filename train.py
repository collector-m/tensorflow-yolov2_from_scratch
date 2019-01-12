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
from core.dataset import dataset, parser

WARM_UP_BATCHES  = 0
BATCH_SIZE       = 1
EPOCHS           = 250 * 1000
LR               = .1e-5
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

parser = parser(IMAGE_H, IMAGE_W, GRID_H, GRID_W, ANCHORS, NUM_CLASSES, DEBUG=False)

trainset = dataset(parser, tfrecord, BATCH_SIZE, shuffle=1)
valset   = dataset(parser, tfrecord, BATCH_SIZE, shuffle=None)

is_training = tf.placeholder(tf.bool)
example = tf.cond(is_training, lambda: trainset.get_next(), lambda: valset.get_next())
input_image, y_true, true_boxes = example

feature_extractor = backbone.FullYoloFeature(input_image, is_training)
features = feature_extractor.feature

output = tf.keras.layers.Conv2D(NUM_ANCHORS * (5 + NUM_CLASSES),
                (1,1), strides=(1,1),
                padding='same',
                name='detection_layer',
                kernel_initializer='lecun_normal')(features)
y_pred = tf.reshape(output, shape=[BATCH_SIZE, GRID_H, GRID_W, NUM_ANCHORS, 5+NUM_CLASSES])

loss_items = utils.compute_loss(y_true, y_pred, true_boxes, GRID_H, GRID_W, BATCH_SIZE, ANCHORS, CLASS_WEIGHTS)
optimizer = tf.train.MomentumOptimizer(LR, momentum=0.9)

update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_op):
    train_op = optimizer.minimize(loss_items[0])

saver = tf.train.Saver(max_to_keep=2)

tf.summary.scalar("loss/loss_xy",      loss_items[1])
tf.summary.scalar("loss/loss_wh",      loss_items[2])
tf.summary.scalar("loss/loss_conf",    loss_items[3])
tf.summary.scalar("loss/loss_class",   loss_items[4])
tf.summary.scalar("yolov2/total_loss", loss_items[0])
tf.summary.scalar("yolov2/recall_50",  loss_items[5])
tf.summary.scalar("yolov2/recall_75",  loss_items[6])
tf.summary.scalar("yolov2/avg_iou",    loss_items[7])

write_op = tf.summary.merge_all()
writer_train = tf.summary.FileWriter("./data/train")
writer_val   = tf.summary.FileWriter("./data/val")

sess.run(tf.global_variables_initializer())
for epoch in range(EPOCHS):
    print("========================> epoch %08d <========================" %epoch)

    _, summary = sess.run([train_op, write_op], feed_dict={is_training:True})
    writer_train.add_summary(summary, global_step=epoch)
    writer_train.flush()

    _, summary = sess.run([loss_items, write_op], feed_dict={is_training:True})
    writer_val.add_summary(summary, global_step=epoch)
    writer_val.flush()

    if epoch % 2000 == 0: saver.save(sess, save_path="./data/checkpoint/yolov2.ckpt", global_step=epoch)



