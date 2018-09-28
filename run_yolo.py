#!/usr/bin/env python
# -*- coding=utf-8 -*-
from __future__ import print_function

import os
import sys
import cv2
import numpy as np
import tensorflow as tf

from utils.utils import load_pb

S = 7
B = 3
Classes = 20

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
    with tf.Session() as sess:
        load_pb(sys.argv[1])
        # tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
        # for tensor_name in tensor_name_list:
        #     print(tensor_name)

        encoded = encode_image(sess, sys.argv[2])

        for i in range(1715):
            print("%.4f " % (encoded[:, i]), end="")
            if (19 == i % 20):
                print("\n", end="")
        
        # SxSx20 P(classi | object) SxSx3 confidence SxSx3x4 bbox
        classes_probs_vector = encoded[:, :S*S*Classes]
        confidences_vector = encoded[:, S*S*Classes : (S*S*Classes+S*S*3)]
        boxes_vector = encoded[:, -S*S*3*4:]
        print(classes_probs_vector.shape)
        print(confidences_vector.shape)
        print(boxes_vector.shape)

        for i in range(49):
            for j in range(3):
                print("grid{} box{} confidence:{} x:{} y:{} w:{}, h{}".format(
                    i, j, confidences_vector[:, i*3+j], boxes_vector[:, i*4*j + 0],
                    boxes_vector[:, i*4*j + 1],
                    boxes_vector[:, i*4*j + 2],
                    boxes_vector[:, i*4*j + 3]
                ))

        # classes_probs = np.reshape(classes_probs_vector, [S, S, Classes])
        # confidences = np.reshape(confidences_vector, [S, S, B])
        # boxes = np.reshape(boxes_vector, [S, S, B, 4])
        # print(classes_probs[0,0,:])
        # print(classes_probs_vector[:, :Classes])
        # print(confidences[0,0,:])
        # print(confidences_vector[:, :B])
        # print(boxes[0,0,:,:])
        # print(boxes_vector[:, :12])

        # class_scores = np.zeros([S, S, B, Classes])
        # for i in range(B):
        #     class_scores[:, :, i, :] = confidences[:, :, i][:,:,np.newaxis] * classes_probs
        #     print(class_scores[:, :, i, :].shape)
        #     # print(class_scores[:, :, i, :])
        # print(class_scores > .2)

        # encode = np.reshape(encode, [S,S,Classes+5*B, 1])
        # print(encode.shape)
        # print(encode[:,:,0])

