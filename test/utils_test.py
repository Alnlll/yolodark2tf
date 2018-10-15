#!/usr/bin/env python
# -*- coding=utf-8 -*-

from __future__ import print_function
# from __future__ import absolute_import

import os
import sys
if not os.path.abspath("../") in sys.path:
    sys.path.append(os.path.abspath("../"))
try:
    from unittest import mock
except ImportError:
    import mock
import unittest

import numpy as np
import tensorflow as tf
from utils.utils import make_unique_section_file
from utils.utils import get_padding_num
from utils.utils import timer_wrapper
from yolo.yolo_utils import select_boxes_by_classes_prob
from yolo.yolo_utils import non_max_suppression
from yolo.yolo_utils import read_classes_names

class TestUtiils(unittest.TestCase):
    # def test_get_padding_num_one_dim(self):
    #     input_shape = [10,13]
    #     kernel_size = [3,3]
    #     stride = 2

    #     self.assertTrue(
    #         (get_padding_num(input_shape, kernel_size, stride) == \
    #         np.array([[1., 2.], [1., 2.]])).all())
    def test_timer_wrapper(self):
        @timer_wrapper
        def func():
            for i in range(1000):
                x = i * i
        func()
class TestYOLOUtils(unittest.TestCase):
    # def test_select_boxes_by_classes_prob(self):
    #     box_confidences = tf.random_normal([3,3,2])
    #     class_probs = tf.random_normal([3,3,5])
    #     boxes = tf.random_normal([3,3,2,4])

    #     print(select_boxes_by_classes_prob(box_confidences, class_probs, boxes))

    # def test_non_max_suppression(self):
    #     box_confidences = tf.random_normal([3,3,2])
    #     class_probs = tf.random_normal([3,3,5])
    #     boxes = tf.random_normal([3,3,2,4])

    #     class_scores,boxes,classes = select_boxes_by_classes_prob(box_confidences, class_probs, boxes)
    #     print(non_max_suppression(class_scores, boxes, classes))

    def test_read_classes_names_voc(self):
        file_path = '../data/voc.names'
        names = read_classes_names(file_path)

        self.assertEqual(
            ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
            names)


if "__main__" == __name__:
    unittest.main()