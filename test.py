#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : test.py
#   Author      : YunYang1994
#   Created date: 2019-01-12 15:38:50
#   Description :
#
#================================================================

import numpy as np
from core import backbone, raw_utils
import cv2
import tensorflow as tf
sess = tf.Session()

image_path = "./2008_003414.jpg"

input_image = tf.placeholder(tf.float32, [1, 416, 416, 3])
input_image = input_image / 255.
anchors  = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
NUM_ANCHORS = len(anchors) // 2
CLASSES          = [ cls.strip('\n') for cls in open("./data/voc.names").readlines()]
NUM_CLASSES      = len(CLASSES)

GRID_H , GRID_W  = 13, 13
IMAGE_H, IMAGE_W = 416, 416

feature_extractor = backbone.FullYoloFeature(input_image, is_training=False)
features = feature_extractor.feature

input_num_filter = features.shape[-1].value
filter_weight = tf.get_variable('weight', [1, 1, input_num_filter, NUM_ANCHORS*(5+NUM_CLASSES)],
                                initializer=tf.truncated_normal_initializer(stddev=1))
filter_weight = filter_weight / (GRID_H*GRID_W)

biases = tf.get_variable('biases', [NUM_ANCHORS*(5+NUM_CLASSES)],
                         initializer=tf.truncated_normal_initializer(stddev=1))
biases = biases / (GRID_H*GRID_W)
output = tf.nn.conv2d(features, filter_weight, strides=[1, 1, 1, 1], padding='SAME')
output = tf.nn.bias_add(output, biases)


y_pred = tf.reshape(output, shape=[1, GRID_H, GRID_W, NUM_ANCHORS, 5+NUM_CLASSES])

image = cv2.imread(image_path)
image_h, image_w, _ = image.shape
image = cv2.resize(image, (416, 416))

saver = tf.train.Saver(max_to_keep=2)
saver.restore(sess, "./data/checkpoint/yolov2.ckpt-0")

net_out = sess.run(y_pred, feed_dict={input_image:np.expand_dims(image, 0)})
net_out = net_out[0]
boxes  = raw_utils.decode_netout(net_out, anchors, 20, 0., 0.0)

image = raw_utils.draw_boxes(image, boxes, CLASSES)

print(len(boxes), 'boxes are found')
cv2.imwrite("./result.jpg", image)

