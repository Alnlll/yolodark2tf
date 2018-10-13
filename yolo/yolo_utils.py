#!/usr/bin/env python
# -*- coding=utf-8 -*-
import os

import numpy as np
import tensorflow as tf

def select_boxes_by_classes_prob(box_confidences, class_probs, boxes, threshold=.2):
    '''
    Select those 
    Arguments:
        box_confidences - tensor of shape [S, S, B], P(object)*IOU(ground truth)
        class_probs - tensor of shape [S, S, C], P(classi | obejct)
        boxes - tensor of shape [S, S, B, 4], coordinates of detected
    Returns:
        class_scores - tensor of shape [S, S, B, C]
    '''
    # S, S, B, _ = box_confidences.shape.as_list()
    _, _, C = class_probs.shape.as_list()

    # Calculate p(classes)
    class_scores = tf.expand_dims(box_confidences, -1) * tf.expand_dims(class_probs, 2)
    class_scores = tf.reshape(class_scores, [-1, C])
    boxes = tf.reshape(boxes, [-1, 4])
    
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
def read_classes_names(file_path, sep='\n'):
    if 'names' != file_path.split(".")[-1]:
        raise TypeError("Need name file end with .names, got {}".format(file_path))
    if not os.path.exists(file_path):
        raise IOError("{} doesn't exist.")
    
    with open(file_path, 'r') as f:
        names = f.read()
        names = names.strip().split(sep)

    return names
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
def activation_func(inputs, activation, leaky_ratio=0.1):
    if 'relu' == activation:
        output = tf.nn.relu(inputs, name = 'relu')
    elif 'leaky' == activation:
        output = tf.nn.leaky_relu(inputs, alpha=leaky_ratio, name = 'leaky')
    elif 'linear' == activation:
        output = inputs
    else:
        raise TypeError("Unknown activation type {}.".format(activation))
    
    return output
def create_bn_layer(
    inputs,
    momentum=0.9,
    eps=1e-5,
    is_training=False,
    name=None):
    '''
    Create batch normalization layer
    '''
    return tf.layers.batch_normalization(
        inputs,
        momentum=momentum,
        epsilon=eps,
        training=is_training,
        name=name)
def create_convolution_layer(
    inputs,
    filters, f_h, f_w,
    stride,
    padding=None,
    batch_norm=True,
    momentum=.9,
    eps=1e-5,
    activation='leaky',
    is_training=False,
    filter_initializer=tf.variance_scaling_initializer(),
    use_bias=False,
    bias_initializer=tf.variance_scaling_initializer(),
    name="conv_layer"):
    '''
    Create a convolution layer.
    Arguments:
    Returns:
    '''

    in_channels = inputs.shape.as_list()[-1]
    padding = 'SAME' if 1 == padding and 1 == stride else 'VALID'

    with tf.variable_scope(name):
        if use_bias:
            output = tf.layers.conv2d(
                inputs,
                filters,
                (f_h, f_w),
                strides=(stride, stride),
                padding=padding,
                use_bias=True,
                kernel_initializer=filter_initializer,
                bias_initializer=bias_initializer)
        else:
            output = tf.layers.conv2d(
                inputs,
                filters,
                (f_h, f_w),
                strides=(stride, stride),
                padding=padding,
                use_bias=False,
                kernel_initializer=filter_initializer)

        if batch_norm:
            output = create_bn_layer(
                output,
                momentum=momentum,
                eps=eps,
                is_training=is_training)
            # output = tf.layers.batch_normalization(
            #     output,
            #     momentum=_BATCH_NORM_DECAY,
            #     epsilon=_BATCH_NORM_EPSILON,
            #     training=is_training)

        # activation
        if activation:
            output = activation_func(output, activation)

        param = {
            'output_shape': output.shape.as_list(),
            'weight_shape': [f_h, f_w, in_channels, filters]}

        return output, param
def create_pooling_layer(
    inputs,
    p_h, p_w,
    pooling_type,
    stride,
    padding="VALID",
    name='poolin_layer'):
    '''
    Pooling layer
    '''
    with tf.name_scope(name):
        if 'avg' == pooling_type:
            output = tf.nn.avg_pool(
                inputs,
                (1, p_h, p_w, 1),
                (1, stride, stride, 1),
                padding,
                name='avg_pooling')
        elif 'max' == pooling_type:
            output = tf.nn.max_pool(
                inputs,
                (1, p_h, p_w, 1),
                (1, stride, stride, 1),
                padding,
                name='avg_pooling')
        else:
            raise TypeError("Required 'avg' or 'max', but received '{}'.".format(pooling_type))
    return output
def create_dropout_layer(
    inputs,
    prob,
    is_training=False,
    seed=None,
    name='dropout_layer'):
    '''
    Dropout Layer.
    '''
    with tf.name_scope(name):
        if is_training:
            output = tf.nn.dropout(inputs, prob, seed=seed, name='dropout')
        else:
            output = inputs

    return output
def create_flatten_layer(inputs, name='flatten_layer'):
    return tf.layers.flatten(inputs, name=name)
def create_fully_connectd_layer(
    inputs,
    n_out,
    activation='leaky',
    activation_fn=None,
    weight_initializer=tf.variance_scaling_initializer(),
    use_bias=True,
    bias_initializer=tf.variance_scaling_initializer(),
    name='fc_layer'):
    '''
    Fully-Connected layer.
    '''

    with tf.variable_scope(name):
        # FC layer
        n_in = inputs.shape.as_list()[-1]
        if use_bias:
            output = tf.contrib.layers.fully_connected(
                inputs,
                n_out,
                activation_fn=activation_fn)
        else:
            output = tf.contrib.layers.fully_connected(
                inputs,
                n_out,
                activation_fn=activation_fn,
                biases_initializer=bias_initializer)

        if activation:
            output = activation_func(output, activation)

        param = {
            'weight_shape': [n_in, n_out],
            'output_shape': output.shape}

        return output, param
