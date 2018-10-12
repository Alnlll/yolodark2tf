#!/usr/bin/env python
# -*- coding=utf-8 -*-
import os

import numpy as np
import tensorflow as tf

def select_boxes_by_classes_prob(box_confidences, class_probs, boxes, threshold=.2):
    '''
    Select those 
    Arguments:
        box_confidences - tensor of shape [S, S, B]
        class_probs - tensor of shape [S, S, C]
        boxes - tensor of shape [S, S, B, 4]
    Returns:
        class_scores - tensor of shape [S, S, B, C]
    '''
    # S, S, B, _ = box_confidences.shape.as_list()
    _, _, C = class_probs.shape.as_list()

    # Calculate p(classes)
    class_scores = tf.expand_dims(box_confidences, -1) * tf.expand_dims(class_probs, 2)
    print(class_scores)
    print(boxes)
    class_scores = tf.reshape(class_scores, [-1, C])
    boxes = tf.reshape(boxes, [-1, 4])

    print(class_scores, boxes)
    
    # Select max p(classes) box btween B boxes for one cell
    box_classes = tf.argmax(class_scores, axis=1)
    box_class_scores = tf.reduce_max(class_scores, axis=1)

    # Select classes' scores by threshold
    mask = box_class_scores >= threshold
    class_scores = tf.boolean_mask(box_class_scores, mask)
    boxes = tf.boolean_mask(boxes, mask)
    box_classes = tf.boolean_mask(box_classes, mask)

    return class_scores, boxes, box_classes
def non_max_suppression(class_scores, boxes, box_classes, max_detect_count=10, iou_threshold=.6):

    # tf nms need coordinates of any diagonal pair of box corner
    # yolo gives center coordinate and h,w, need thranslate
    nms_boxes = tf.stack([
        boxes[:, 1] - boxes[:, 2]/2, 
        boxes[:, 0] - boxes[:, 3]/2,
        boxes[:, 1] + boxes[:, 2]/2,
        boxes[:, 0] + boxes[:, 3]/2],
        axis=1)

    indices = tf.image.non_max_suppression(
        nms_boxes, class_scores,
        max_detect_count, iou_threshold)
    
    scores = tf.identity(tf.gather(class_scores, indices))
    boxes = tf.identity(tf.gather(boxes, indices))#(xcenter,ycenter,w,h)
    classes = tf.identity(tf.gather(box_classes, indices))

    return scores, boxes, classes
def read_classes_names(file_path):
    if 'names' != file_path.split(".")[-1]:
        raise TypeError("Need name file end with .names, got {}".format(file_path))
    if not os.path.exists(file_path):
        raise IOError("{} doesn't exist.")
    
    with open(file_path, 'r') as f:
        names = f.read()
        print(names)
# def IOU(box0, box1):
#     b1_x0, b1_y0, b1_x1, b1_y1 = box0
#     b2_x0, b2_y0, b2_x1, b2_y1 = box1

#     x0 = max(b1_x0, b2_x0)
#     y0 = max(b1_y0, b2_y0)
#     x1 = min(b1_x1, b2_x1)
#     y1 = min(b1_y1, b2_y1)

#     inter_area = (x1 - x0) * (y1 - y0)
#     b1_area = (b1_x1 - b1_x0) * (b1_y0 - b1_y1)
#     b2_area = (b2_x1 - b2_x0) * (b2_y0 - b2_y1)

#     iou = inter_area / (b1_area + b2_area - inter_area + 1e-05)

#     return iou
