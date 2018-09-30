#!/usr/bin/env python
# -*- coding=utf-8 -*-

from __future__ import print_function
# from __future__ import absolute_import

import os
import sys
if not os.path.abspath("../") in sys.path:
    sys.path.append(os.path.abspath("../"))

from utils.utils import make_unique_section_file

def test_make_unique_section_file():
    make_unique_section_file("../cfg/yolov1.cfg")

test_list = [
    test_make_unique_section_file
]

if "__main__" == __name__:
    for test_item in test_list:
        test_item()