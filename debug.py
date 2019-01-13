#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : debug.py
#   Author      : YunYang1994
#   Created date: 2019-01-13 03:07:09
#   Description :
#
#================================================================

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from core.dataset import dataset, parser

WARM_UP_BATCHES  = 0
BATCH_SIZE       = 1
EPOCHS           = 250 * 1000
LR               = .5e-4
GRID_H , GRID_W  = 13, 13
IMAGE_H, IMAGE_W = 416, 416
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
NUM_ANCHORS      = len(ANCHORS) // 2
CLASSES          = [ cls.strip('\n') for cls in open("./data/voc.names").readlines()]
NUM_CLASSES      = len(CLASSES)
CLASS_WEIGHTS    = np.ones(NUM_CLASSES, dtype='float32')
TRUE_BOX_BUFFER  = 20

tfrecord = "./test_data/train0003.tfrecords"
sess = tf.Session()

parser = parser(IMAGE_H, IMAGE_W, GRID_H, GRID_W, ANCHORS, NUM_CLASSES, DEBUG=True)

trainset = dataset(parser, tfrecord, BATCH_SIZE, shuffle=1)

example = trainset.get_next()


for l in range(2):
    image, boxes = sess.run(example)
    image, boxes = image[0], boxes[0]

    n_box = len(boxes)
    for i in range(n_box):
        image = cv2.rectangle(image,(int(float(boxes[i][0])),
                                    int(float(boxes[i][1]))),
                                    (int(float(boxes[i][2])),
                                    int(float(boxes[i][3]))), (255,0,0), 1)

    image = Image.fromarray(np.uint8(image))
    image.show()




