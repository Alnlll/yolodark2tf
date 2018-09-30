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

from utils.utils import make_unique_section_file
from utils.utils import get_padding_num

def test_make_unique_section_file():
    make_unique_section_file("../cfg/yolov1.cfg")

class TestUtils(unittest.TestCase):
    def test_get_padding_num_one_dim(self):
        input_shape = [10,13]
        kernel_size = [3,3]
        stride = 2

        self.assertTrue(
            (get_padding_num(input_shape, kernel_size, stride) == \
            np.array([[1., 2.], [1., 2.]])).all())

if "__main__" == __name__:
    unittest.main()