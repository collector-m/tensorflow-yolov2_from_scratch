#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : backbone.py
#   Author      : YunYang1994
#   Created date: 2019-01-12 13:49:33
#   Description :
#
#================================================================

import tensorflow as tf


class BaseFeatureExtractor(object):
    """docstring for ClassName"""

    # to be defined in each subclass
    def __init__(self, input_image):
        raise NotImplementedError("error message")

    # to be defined in each subclass
    def load_weights(self, weights):
        raise NotImplementedError("error message")

    def get_output_shape(self):
        return self.feature.shape.as_list()[1:3]

class TinyYoloFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_image, is_training=True):

        # Layer 1
        x = tf.keras.layers.Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2 - 5
        for i in range(4):
            x = tf.keras.layers.Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', name='conv_' + str(i+2), use_bias=False)(x)
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.leaky_relu(x, alpha=0.1)
            x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = tf.keras.layers.Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

        # Layer 7 - 8
        for i in range(2):
            x = tf.keras.layers.Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_' + str(i+7), use_bias=False)(x)
            x = tf.layers.batch_normalization(x, training=is_training)
            x = tf.nn.leaky_relu(x, alpha=0.1)

        self.feature = x


class FullYoloFeature(BaseFeatureExtractor):
    """docstring for ClassName"""
    def __init__(self, input_image, is_training=True):
        """
        Full yolov2 feature extractor network
        """
        space_to_depth_x2 = lambda x: tf.space_to_depth(x, block_size=2)

        # Layer 1
        x = tf.keras.layers.Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2
        x = tf.keras.layers.Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 3
        x = tf.keras.layers.Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        # Layer 4
        x = tf.keras.layers.Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        # Layer 5
        x = tf.keras.layers.Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = tf.keras.layers.Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        # Layer 7
        x = tf.keras.layers.Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        # Layer 8
        x = tf.keras.layers.Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 9
        x = tf.keras.layers.Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        # Layer 10
        x = tf.keras.layers.Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        # Layer 11
        x = tf.keras.layers.Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        # Layer 12
        x = tf.keras.layers.Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        # Layer 13
        x = tf.keras.layers.Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        skip_connection = x
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = tf.keras.layers.Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        # Layer 15
        x = tf.keras.layers.Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        # Layer 16
        x = tf.keras.layers.Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        # Layer 17
        x = tf.keras.layers.Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        # Layer 18
        x = tf.keras.layers.Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        # Layer 19
        x = tf.keras.layers.Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        # Layer 20
        x = tf.keras.layers.Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        # Layer 21
        skip_connection = tf.keras.layers.Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        skip_connection = tf.keras.layers.Lambda(space_to_depth_x2)(skip_connection)

        x = tf.keras.layers.concatenate([skip_connection, x])

        # Layer 22
        x = tf.keras.layers.Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
        x = tf.layers.batch_normalization(x, training=is_training)
        x = tf.nn.leaky_relu(x, alpha=0.1)

        self.feature = x




