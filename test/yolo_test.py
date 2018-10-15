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

import cv2
import numpy as np
import tensorflow as tf

from yolo.yolov1 import Yolov1

class TestYolov1(unittest.TestCase):

    # def test_yolov1_create_conv_layer_valid_pad(self):
    #     flags = mock.Mock()
    #     flags.cfg = "../cfg/yolov1.cfg"
    #     flags.train = True

    #     model = Yolov1(flags)
    #     x = tf.random_normal([1,10,10,3])
    #     y = model.create_convolution_layer(
    #         x,
    #         64, 3, 3,
    #         2,
    #         0,
    #         1,
    #         'leaky',
    #         name="test_convlayer1"
    #     )
    #     self.assertEqual(
    #         y.shape,
    #         tf.TensorShape([tf.Dimension(1), tf.Dimension(4), tf.Dimension(4), tf.Dimension(64)]))

    # def test_yolov1_create_conv_layer_same_pad(self):
    #     flags = mock.Mock()
    #     flags.cfg = "../cfg/yolov1.cfg"
    #     flags.train = True

    #     model = Yolov1(flags)
    #     x = tf.random_normal([1,10,10,3])
    #     y = model.create_convolution_layer(
    #         x,
    #         64, 3, 3,
    #         1,
    #         1,
    #         1,
    #         'leaky',
    #         name="test_convlayer2"
    #     )
    #     self.assertEqual(
    #         y.shape,
    #         tf.TensorShape([tf.Dimension(1), tf.Dimension(10), tf.Dimension(10), tf.Dimension(64)]))
    
    # def test_yolov1_create_model(self):
    #     flags = mock.Mock()
    #     flags.cfg = "../cfg/yolov1.cfg"
    #     flags.train = True
        
    #     model = Yolov1(flags)
    #     model.load_model()

    # def test_yolov1_load_weight_file_not_exists(self):
        # flags = mock.Mock()
        # flags.weight = "weights/yolov1.weight"
        # model = Yolov1(flags)
        # self.assertRaises(IOError, model.load_weight)
    
    # def test_yolov1_load_weight_header(self):
    #     flags = mock.Mock()
    #     flags.cfg = "../cfg/yolov1.cfg"
    #     flags.weight = "../weights/yolov1.weights"
    #     flags.train = False
    #     model = Yolov1(flags)
    #     model.load_model()
    #     model.load_weight()

    # def test_yolov1_detect(self):
    #     flags = mock.Mock()
    #     flags.cfg = "../cfg/yolov1.cfg"
    #     flags.weight = "../weights/yolov1.weights"
    #     flags.train = False
    #     image = cv2.imread("../data/dog.jpg")
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     image = cv2.resize(image, (448, 448))
    #     image = image.astype(np.float32)
    #     image = image.reshape([1,448,448,3])

    #     # print(image.shape, image.dtype)
    #     # return

    #     model = Yolov1(flags)
    #     model.load_model()
    #     model.load_weight()
    #     print(model.detect(image))

    def test_yolov1_detect_yolov1_tiny(self):
        flags = mock.Mock()
        flags.cfg = "../cfg/yolov1-tiny.cfg"
        flags.weight = "../weights/tiny-yolov1.weights"
        flags.train = False
        flags.summary = "summary"
        flags.names = '../data/voc.names'

        # image = cv2.imread("../data/dog.jpg")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (448, 448))
        # image = image.astype(np.float32)
        # image = image.reshape([1,448,448,3])

        # print(image.shape, image.dtype)
        # return

        model = Yolov1(flags)
        # model._load_model()
        # model._load_weight()
        model.detect_from_image_file("../data/person.jpg")

if "__main__" == __name__:
    unittest.main()