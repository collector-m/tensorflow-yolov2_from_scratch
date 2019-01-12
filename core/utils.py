#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : tf_utils.py
#   Author      : YunYang1994
#   Created date: 2019-01-12 14:12:21
#   Description :
#
#================================================================

import numpy as np
import tensorflow as tf

def compute_loss(y_true, y_pred, true_boxes, grid_h, grid_w, batch_size, anchors, class_weights, warm_up_batches=0):
    """
    compute loss
    """
    NO_OBJECT_SCALE  = 1.0
    OBJECT_SCALE     = 5.0
    COORD_SCALE      = 1.0
    CLASS_SCALE      = 1.0
    NUM_ANCHORS      = len(anchors) // 2

    mask_shape = tf.shape(y_true)[:4]
    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1, 1)))
    cell_y = tf.transpose(cell_x, (0,2,1,3,4))
    cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [batch_size, 1, 1, 5, 1])

    coord_mask = tf.zeros(mask_shape)
    conf_mask  = tf.zeros(mask_shape)
    class_mask = tf.zeros(mask_shape)

    seen = tf.Variable(0.)

    """
    Adjust prediction
    """
    ### adjust x and y
    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

    ### adjust w and h
    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(anchors, [1,1,1,NUM_ANCHORS,2])

    ### adjust confidence
    pred_box_conf = tf.sigmoid(y_pred[..., 4])

    ### adjust class probabilities
    pred_box_class = y_pred[..., 5:]

    """
    Adjust ground truth
    """
    ### adjust x and y
    true_box_xy = y_true[..., 0:2] # relative position to the containing cell

    ### adjust w and h
    true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically

    ### adjust confidence
    true_wh_half = true_box_wh / 2.
    true_mins    = true_box_xy - true_wh_half
    true_maxes   = true_box_xy + true_wh_half

    pred_wh_half = pred_box_wh / 2.
    pred_mins    = pred_box_xy - pred_wh_half
    pred_maxes   = pred_box_xy + pred_wh_half

    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)

    true_box_conf = iou_scores * y_true[..., 4]
    ### adjust class probabilities
    true_box_class = tf.argmax(y_true[..., 5:], -1)

    """
    Determine the masks
    """
    ### coordinate mask: simply the position of the ground truth boxes (the predictors)
    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE

    ### confidence mask: penelize predictors + penalize boxes with low IOU
    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.5
    true_xy = true_boxes[..., 0:2]
    true_wh = true_boxes[..., 2:4]

    true_wh_half = true_wh / 2.
    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half

    pred_xy = tf.expand_dims(pred_box_xy, 4)
    pred_wh = tf.expand_dims(pred_box_wh, 4)

    pred_wh_half = pred_wh / 2.
    pred_mins    = pred_xy - pred_wh_half
    pred_maxes   = pred_xy + pred_wh_half

    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = tf.truediv(intersect_areas, union_areas)

    best_ious = tf.reduce_max(iou_scores, axis=4)
    conf_mask = conf_mask + tf.to_float(best_ious < 0.5) * (1 - y_true[..., 4]) * NO_OBJECT_SCALE

    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
    conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE
    ### class mask: simply the position of the ground truth boxes (the predictors)
    class_mask = y_true[..., 4] * tf.gather(class_weights, true_box_class) * CLASS_SCALE

    """
    Warm-up training
    """
    no_boxes_mask = tf.to_float(coord_mask < COORD_SCALE/2.)
    seen = tf.assign_add(seen, 1.)

    true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, warm_up_batches),
                          lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                   true_box_wh + tf.ones_like(true_box_wh) * np.reshape(anchors, [1,1,1,NUM_ANCHORS,2]) * no_boxes_mask,
                                   tf.ones_like(coord_mask)],
                          lambda: [true_box_xy,
                                   true_box_wh,
                                   coord_mask])
    """
    Finalize the loss
    """

    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

    loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

    total_loss = loss_xy + loss_wh + loss_conf + loss_class

    nb_true_box = tf.reduce_sum(y_true[..., 4])
    nb_pred_50  = tf.reduce_sum(tf.to_float(true_box_conf > 0.50) * tf.to_float(pred_box_conf > 0.3))
    nb_pred_75  = tf.reduce_sum(tf.to_float(true_box_conf > 0.75) * tf.to_float(pred_box_conf > 0.3))
    """
    caculate recall and iou
    """
    recall_50 = nb_pred_50 / (nb_true_box + 1e-6)
    recall_75 = nb_pred_75 / (nb_true_box + 1e-6)
    """
    Debugging code
    """
    total_loss = tf.Print(total_loss, [loss_xy],     message='loss  xy    =\t', summarize=1000)
    total_loss = tf.Print(total_loss, [loss_wh],     message='loss  wh    =\t', summarize=1000)
    total_loss = tf.Print(total_loss, [loss_conf],   message='loss  conf  =\t', summarize=1000)
    total_loss = tf.Print(total_loss, [loss_class],  message='loss  class =\t', summarize=1000)
    total_loss = tf.Print(total_loss, [total_loss],  message='total loss  =\t', summarize=1000)
    total_loss = tf.Print(total_loss, [recall_50],   message='recall_50   =\t', summarize=1000)
    total_loss = tf.Print(total_loss, [recall_75],   message='recall_50   =\t', summarize=1000)

    return total_loss, loss_xy, loss_wh, loss_conf, loss_class, recall_50, recall_75




def resize_image_correct_bbox(image, boxes, image_h, image_w):

    image_size = tf.to_float(tf.shape(image)[0:2])[::-1]
    image = tf.image.resize_images(image, size=[image_h, image_w])

    # correct bbox
    xx1 = boxes[:, 0] * image_h / image_size[0]
    yy1 = boxes[:, 1] * image_w / image_size[1]
    xx2 = boxes[:, 2] * image_h / image_size[0]
    yy2 = boxes[:, 3] * image_w / image_size[1]
    idx = boxes[:, 4]

    boxes = tf.stack([xx1, yy1, xx2, yy2, idx], axis=1)
    return image, boxes

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
        image, gt_boxes = resize_image_correct_bbox(image, gt_boxes, self.IMAGE_H, self.IMAGE_W)
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
            print(box)
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
                iou = intersect_area / (box_area + anchors_area - intersect_area)
                best_anchor = np.argmax(iou)
                y_batch[grid_y, grid_x, best_anchor, 0:4      ] = box
                y_batch[grid_y, grid_x, best_anchor, 4        ] = 1.
                y_batch[grid_y, grid_x, best_anchor, 5+cls_idx] = 1.

                b_batch[0, 0, 0, i] = box

        return y_batch, b_batch

# continue to work


def reorg_layer(netout, anchors, num_classes, obj_threshold=0.3, nms_threshold=0.3):

    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []

    # decode the output by the network
    netout[..., 4]  = _sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row,col,b,5:]

                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row,col,b,:4]

                    x = (col + _sigmoid(x)) / grid_w # center position, unit: image width
                    y = (row + _sigmoid(y)) / grid_h # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                    confidence = netout[row,col,b,4]

                    box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)

                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(num_classes):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]

    return boxes

    pass

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x/np.min(x)*t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin

    union = w1*h1 + w2*h2 - intersect

    return float(intersect) / union

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation, first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3
