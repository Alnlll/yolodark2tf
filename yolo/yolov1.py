#!/usr/bin/env python
#-*- coding=utf-8 -*-

from __future__ import print_function

import os
try:
    import ConfigParser as configparser
except ImportError:
    import configparser

from utils.utils import make_unique_section_file
import tensorflow as tf

class DarkNet(object):
    def __init__(self):
        self.cfg = configparser.ConfigParser()
        self.cfg_sections = None
    def __del__(self):
        pass
    def parse_config(self, cfg_path):
        if not os.path.exists(cfg_path):
            raise IOError("{} doesn't exists.".format(cfg_path))
        # Get cfg file with unique section name
        uni_cfg_path, self.cfg_sections = make_unique_section_file(cfg_path)
        # Read cfg file
        self.cfg.read(uni_cfg_path)

        # Get input image info
        try:
            self.img_height = self.cfg.getint(self.cfg_sections[0], "height")
            self.img_width = self.cfg.getint(self.cfg_sections[0], "width")
            self.img_channels = self.cfg.getint(self.cfg_sections[0], "channels")
        except Exception as e:
            raise ValueError(e)
        
        # Get output feature map info
        try:
            self.classes = self.cfg.getint(self.cfg_sections[-1], "classes")
            self.cell_size = self.cfg.getint(self.cfg_sections[-1], "side")
            self.box_nums = self.cfg.getint(self.cfg_sections[-1], "num")
        except Exception as e:
            raise ValueError(e)
        
        # print(self.img_height, self.img_width, self.img_channels)
        # print(self.classes, self.cell_size, self.box_nums)

    def construct_model(self):
        pass
    def load_weight(self, weight_path):
        pass;