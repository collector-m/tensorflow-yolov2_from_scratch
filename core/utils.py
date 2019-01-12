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
    nb_iou_box   = tf.reduce_sum(best_ious * tf.to_float(y_true[..., 4] > 0.0))

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
    avg_iou   = nb_iou_box / (nb_true_box + 1e-6)
    """
    Debugging code
    """
    total_loss = tf.Print(total_loss, [loss_xy],     message='loss  xy    =\t', summarize=1000)
    total_loss = tf.Print(total_loss, [loss_wh],     message='loss  wh    =\t', summarize=1000)
    total_loss = tf.Print(total_loss, [loss_conf],   message='loss  conf  =\t', summarize=1000)
    total_loss = tf.Print(total_loss, [loss_class],  message='loss  class =\t', summarize=1000)
    total_loss = tf.Print(total_loss, [total_loss],  message='total loss  =\t', summarize=1000)
    total_loss = tf.Print(total_loss, [recall_50],   message='recall_50   =\t', summarize=1000)
    total_loss = tf.Print(total_loss, [recall_75],   message='recall_75   =\t', summarize=1000)
    total_loss = tf.Print(total_loss, [avg_iou],     message='avg_iou     =\t', summarize=1000)

    return total_loss, loss_xy, loss_wh, loss_conf, loss_class, recall_50, recall_75, avg_iou



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


