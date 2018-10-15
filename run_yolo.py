#!/usr/bin/env python
# -*- coding=utf-8 -*-
from __future__ import print_function

import os
import sys
import cv2
import argparse
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
import tensorflow as tf

from utils.utils import load_pb
from yolo.yolov1 import Yolov1
from yolo.yolov2 import Yolov2

def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", action="store_true", default=False,
        help="Run as train mode(test mode if not set).",)
    # parser.add_argument(
    #     "--test", action="store_true", default=True,
    #     help="Run as test mode.")
    parser.add_argument(
        "--cfg", '-c', type=str, default="cfg/yolo.cfg",
        help="Path to config file for network in darknet type.")
    parser.add_argument(
        "--weight", '-w', type=str, default="weights/yolo.weight",
        help="Path to weight file for network in darknet type.")
    parser.add_argument(
        "--image", '-i', type=str, default="data/test.jpg",
        help="Path to image to be detected.")
    parser.add_argument(
        "--video", '-v', type=str, default="data/video.avi",
        help="Path to video to be detected.")
    parser.add_argument(
        "--img_size", type=int, default=448,
        help="Excepted size of image feeded to detect network, invalid when 1 == version.")
    parser.add_argument(
        "--output_dir", '-o', type=str, default="output",
        help="Path to store output.")
    parser.add_argument(
        "--text_record", '-t', type=str, default="",
        help="Text file to restore detect results.")
    parser.add_argument(
        "--summary", '-s', type=str, default="",
        help="Path of summary logs."
    )
    parser.add_argument(
        "--names", '-n', type=str, default="",
        help="Path of class names file."
    )
    parser.add_argument(
        "--version", type=int, default=1,
        help="Version of YOLO to be used."
    )
    parser.add_argument(
        "--score_thresh", type=float, default=.2,
        help="Theshold to filter boxes with low confidence."
    )
    parser.add_argument(
        "--iou_thresh", type=float, default=.5,
        help="Threshold to filter overlaped boxes."
    )
    
    args = parser.parse_args()

    if args.train and args.test:
        print("train and test flag cannot be set simultaneously.")
        sys.exit()

    return args

if "__main__" == __name__:
    # models = [Yolov1, Yolov2, Yolov3]
    models = [Yolov1, Yolov2]
    args = define_args()
    model = models[args.version - 1](args)
    model.detect_from_image_file(args.image)
