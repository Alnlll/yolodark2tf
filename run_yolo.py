#!/usr/bin/env python
# -*- coding=utf-8 -*-
from __future__ import print_function

import os
import sys
import cv2
import argparse
import numpy as np
import tensorflow as tf

from utils.utils import load_pb
from yolo.yolov1 import DarkNet

def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", action="store_true", default=False,
        help="Run as train mode.",)
    parser.add_argument(
        "--test", action="store_true", default=True,
        help="Run as test mode.")
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
        "--output_dir", '-o', type=str, default="output",
        help="Path to store output.")
    parser.add_argument(
        "--summary", '-s', type=str, default="",
        help="Path of summary logs."
    )
    parser.add_argument(
        "--names", '-n', type=str, default="",
        help="Path of class names file."
    )
    
    args = parser.parse_args()

    if args.train and args.test:
        print("train and test flag cannot be set simultaneously.")
        sys.exit()

    return args

if "__main__" == __name__:

    args = define_args()
    model = DarkNet(args)
    model.create_model()

    # with tf.Session() as sess:
    #     load_pb(sys.argv[1])
        
        # SxSx20 P(classi | object) SxSx3 confidence SxSx3x4 bbox
        # classes_probs_vector = encoded[:, :S*S*Classes]
        # confidences_vector = encoded[:, S*S*Classes : (S*S*Classes+S*S*3)]
        # boxes_vector = encoded[:, -S*S*3*4:]
        # print(classes_probs_vector.shape)
        # print(confidences_vector.shape)
        # print(boxes_vector.shape)
