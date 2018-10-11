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
        help="Path to vedio to be detected.")
    
    args = parser.parse_args()

    if args.train and args.test:
        print("train and test flag cannot be set simultaneously.")
        sys.exit()

    print(args)
    return args

def select_classes(box_confidences, class_probs, boxes, threshold=.2):
    '''
    Select those 
    Arguments:
        box_confidences - tensor of shape [S, S, B]
        class_probs - tensor of shape [S, S, C]
        boxes - tensor of shape [S, S, B, 4]
    Returns:
        class_scores - tensor of shape [S, S, B, C]
    '''
    _, _, B = box_confidences.shape


    class_scores = np.zeros([S, S, B, Classes])
    for i in range(B):
        class_scores[:, :, i, :] = confidences[:, :, i][:,:,np.newaxis] * classes_probs
        print(class_scores[:, :, i, :].shape)
    
    print(np.max(class_scores))
    print(tf.keras.backend.max(class_scores, axis=-1))

    return class_scores

def encode_image(sess, image_path):
    
    input = sess.graph.get_tensor_by_name("input:0")
    output = sess.graph.get_tensor_by_name("output:0")

    img = cv2.imread(sys.argv[2])
    img = cv2.resize(img, (448, 448))
    img = img[np.newaxis,:, :, :]

    encode = sess.run(output, feed_dict={input:img})
    print(encode.shape)

    return encode

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
