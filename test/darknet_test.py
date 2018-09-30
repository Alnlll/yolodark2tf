#!/usr/bin/env python
# -*- coding=utf-8 -*-

from __future__ import print_function
# from __future__ import absolute_import

import os
import sys
if not os.path.abspath("../") in sys.path:
    sys.path.append(os.path.abspath("../"))

try:
    import ConfigParser as configparser
except ImportError:
    import configparser
try:
    from unittest import mock
except ImportError:
    import mock
import unittest

import numpy as np
import tensorflow as tf

from yolo.yolov1 import DarkNet

def test_darknet_parse_cfg():
    model = DarkNet()
    print(model.parse_config('../cfg/yolov1.cfg'))

def test_darknet_create_model():
    flags = mock.Mock()
    model = DarkNet(None)
    model.create_model('../cfg/yolov1.cfg')

def test_dark_create_bn_layer():
    model = DarkNet(None)

    sess = tf.InteractiveSession()
    x = tf.constant([[1,2,3],[2,4,8],[3,9,27]],dtype=tf.float32)
    y = model.create_bn_layer(x, is_training=True)
    sess.run(tf.global_variables_initializer())
    print(x.eval())
    print(y.eval())
    sess.close()

def test_dark_create_local_convolution_layer():
    model = DarkNet(None)

    sess = tf.InteractiveSession()
    x = tf.random_normal([1,2,2,3])
    y = model.create_local_convolution_layer(
        x,
        16, 1, 1,
        1,
        1,
        'leaky',
        name='test')
    sess.run(tf.global_variables_initializer())
    print(x.eval())
    print(y.eval())
    print(y.shape)
    print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    sess.close()

def test_dark_create_pooling_layer():
    model = DarkNet(None)

    x = tf.random_normal([1,10,10,3])
    y1 = model.create_pooling_layer(
        x,
        2, 2,
        'max',
        2,
        name='test1')
    y2 = model.create_pooling_layer(
        x,
        2, 2,
        'avg',
        2,
        name='test2')
    print(y1)
    print(y2)

def test_dark_create_fc_layer():
    model = DarkNet(None)

    x = tf.random_normal([1,10,10,3])
    y1 = model.create_fully_connectd_layer(
        x,
        10,
        name='test_fc')
    print(y1)

def test_darknet_create_conv_layer():
    model = DarkNet(None)

    input = tf.placeholder(tf.float32, shape = [2, 120, 120, 3], name = 'input')
    print(model.create_convolution_layer(
        input,
        10, 3, 3,
        2,
        1,
        0,
        'relu',
        name='test'
        ))

def test_darknet_create_dropout_layer():
    model = DarkNet(None)
    x = tf.random_normal([2,3,4])
    y1 = model.create_dropout_layer(x, 0.5, name='test_dropout1')
    print(y1)
    y2 = model.create_dropout_layer(x, 0.5, is_training=True, name='test_dropout2')
    print(y2)
test_list = [
    # test_darknet_parse_cfg,
    # test_darknet_create_model,
    # test_darknet_create_conv_layer,
    # test_dark_create_local_convolution_layer,
    # test_dark_create_pooling_layer,
    # test_dark_create_fc_layer,
    # test_darknet_create_dropout_layer,
    # test_dark_create_bn_layer
    test_darknet_create_model
]

class TestDarkNet(unittest.TestCase):

    def test_darknet_create_conv_layer_valid_pad(self):
        flags = mock.Mock()
        flags.cfg = "../cfg/yolov1.cfg"
        flags.train = True

        model = DarkNet(flags)
        x = tf.random_normal([1,10,10,3])
        y = model.create_convolution_layer(
            x,
            64, 3, 3,
            2,
            0,
            1,
            'leaky',
            name="test_convlayer1"
        )
        self.assertEqual(
            y.shape,
            tf.TensorShape([tf.Dimension(1), tf.Dimension(4), tf.Dimension(4), tf.Dimension(64)]))

    def test_darknet_create_conv_layer_same_pad(self):
        flags = mock.Mock()
        flags.cfg = "../cfg/yolov1.cfg"
        flags.train = True

        model = DarkNet(flags)
        x = tf.random_normal([1,10,10,3])
        y = model.create_convolution_layer(
            x,
            64, 3, 3,
            1,
            1,
            1,
            'leaky',
            name="test_convlayer2"
        )
        self.assertEqual(
            y.shape,
            tf.TensorShape([tf.Dimension(1), tf.Dimension(10), tf.Dimension(10), tf.Dimension(64)]))
    
    def test_darknet_create_model(self):
        flags = mock.Mock()
        flags.cfg = "../cfg/yolov1.cfg"
        flags.train = True
        
        model = DarkNet(flags)
        model.create_model()

if "__main__" == __name__:
    unittest.main()