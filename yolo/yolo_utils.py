#!/usr/bin/env python
# -*- coding=utf-8 -*-
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf

def select_boxes_by_classes_prob(box_confidences, class_probs, boxes, threshold=.2):
    '''
    Select those 
    Arguments:
        box_confidences - tensor of shape [None, S, S, B], P(object)*IOU(ground truth)
        class_probs - tensor of shape [None, S, S, C] or [None, S, S, B, C], P(classi | obejct)
        boxes - tensor of shape [None, S, S, B, 4], coordinates of detected
    Returns:
        class_scores - tensor of shape [S, S, B, C]
    '''
    S, S, B, _ = box_confidences.shape.as_list()
    C = class_probs.shape.as_list()[-1]

    # Calculate p(classes)
    if 4 == len(class_probs.shape):
        class_scores = tf.expand_dims(box_confidences, -1) * tf.expand_dims(class_probs, 3)
        class_scores = tf.reshape(class_scores, [-1, C])
    elif 5 == len(class_probs.shape):
        class_scores = tf.expand_dims(box_confidences, -1) * class_probs
        class_scores = tf.reshape(class_scores, [-1, C])
    else:
        raise ValueError("need 4,5 ndims class_probs, got {}".format(class_probs))
    
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
    '''
    Select box by confidence
    Arguments:
        box_confidences - tensor of shape [None, H*W, B, 1], P(object)*IOU(ground truth)
        class_probs - tensor of shape [None, H*W, C] or [None, H*W, B, C], P(classi | obejct)
        boxes - tensor of shape [None, h*W B, 4], coordinates of detected
    Returns:
        class_scores - tensor of shape [None, H*W, B, C]
    '''
    # C = class_probs.shape.as_list()[-1]

    # # Calculate p(classes)
    # if 3 == len(class_probs.shape):
    #     class_scores = box_confidences * tf.expand_dims(class_probs, 2)
    #     class_scores = tf.reshape(class_scores, [-1, C])
    # elif 4 == len(class_probs.shape):
    #     class_scores = box_confidences * class_probs
    #     print("scores:\n", class_scores)
    #     print("C:\n", C)
    #     class_scores = tf.reshape(class_scores, [-1, C])
    # else:
    #     raise ValueError("need 4,5 ndims class_probs, got {}".format(class_probs))
    
    # boxes = tf.reshape(boxes, [-1, 4])
    
    # # Select max p(classes) box btween B boxes for one cell
    # box_classes = tf.argmax(class_scores, axis=1)
    # box_class_scores = tf.reduce_max(class_scores, axis=1)

    # # Select classes' scores by threshold
    # mask = box_class_scores >= threshold
    # class_scores = tf.boolean_mask(box_class_scores, mask)
    # boxes = tf.boolean_mask(boxes, mask)
    # box_classes = tf.boolean_mask(box_classes, mask)

    # return class_scores, boxes, box_classes
def non_max_suppression(class_scores, boxes, box_classes, max_detect_count=10, iou_threshold=.5):
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
def activation_func(inputs, activation, leaky_ratio=0.1):
    if 'relu' == activation:
        output = tf.nn.relu(inputs, name = 'relu')
    elif 'leaky' == activation:
        output = tf.nn.leaky_relu(inputs, alpha=leaky_ratio, name='leaky')
    elif 'linear' == activation:
        output = inputs
    else:
        raise TypeError("Unknown activation type {}.".format(activation))
    
    return output
# Found from https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
def fixed_padding(inputs, kernel_size, data_format='channels_last', mode='CONSTANT'):
  """Pads the input along the spatial dimensions independently of input size.
  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer list of lenght 2.
    data_format: The input format ('channels_last' or 'channels_first').
    mode: The mode for tf.pad.
  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total_h, pad_total_w = kernel_size[0] - 1, kernel_size[1] - 1
  pad_beg_h, pad_beg_w = pad_total_h // 2, pad_total_w // 2
  pad_end_h, pad_end_w = pad_total_h - pad_beg_h, pad_total_w - pad_beg_w

  if data_format == 'channels_first':
    padded_inputs = tf.pad(
        inputs,
        [[0, 0], [0, 0], [pad_beg_h, pad_end_h], [pad_beg_w, pad_end_w]],
        mode=mode)
  else:
    padded_inputs = tf.pad(
        inputs, 
        [[0, 0], [pad_beg_h, pad_end_h], [pad_beg_w, pad_end_w], [0, 0]],
        mode=mode)
  return padded_inputs
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

    if 1 < stride:
        inputs = fixed_padding(inputs, (f_h, f_w))

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
                name='max_pooling')
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
# Found from https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe
def create_upsample_layer(inputs, factor, name='upsampling_layer'):
    if 4 != len(inputs.shape):
        raise ValueError("Need inputs of 4-D tensor, got {}".format(inputs))

    input_size = inputs.shape.as_list()[1:3]
    output_size = [size*factor for size in input_size]
    h, w = output_size

    with tf.name_scope(name):
        # Need 1 pixel padded, so kernel_size=(3,3)
        inputs = fixed_padding(inputs, (3,3), mode='SYMMETRIC')

        # Because 1 pixel padded at front and end.
        upsample_h, upsample_w = h + 2*factor, w + 2*factor

        # Do upsampling
        output = tf.image.resize_bilinear(inputs, (upsample_h, upsample_w))

        # Split padded region
        output = output[:, 2:-2, 2:-2, :]

    return output
def create_v3_detection_layer(
    inputs, classes, anchors,
    img_size=(416,416),
    use_softmax=True,
    name='detection_layer'):
    B, C = len(anchors), classes
    _, H, W, _ = inputs.shape.as_list()
    img_height, img_width = img_size

    with tf.name_scope(name):
        detection = tf.reshape(inputs, [-1, H, W, B, C + 5])

        # bx = sigmoid(tx) + cx, by = sigmoid(ty) + cy, shape:[None, H, W, B, 2]
        xy_offsets = tf.nn.sigmoid(detection[:,:,:,:,0:2], name='xy_offsets')
        # bw = pw * exp(tw), bh = ph * exp(th), shape:[None, H, W, B, 2]
        wh_scales = tf.exp(detection[:,:,:,:,2:4], name='wh_scales')
        # confidence = sigmoid(to), shape:[None, H, W,, B]
        box_confidences = tf.nn.sigmoid(detection[:,:,:,:,4], name='box_confidences')
        
        if use_softmax:
            class_probs = tf.nn.softmax(detection[:,:,:,:,5:], name="class_probs")
        else:
            class_probs = detection[:,:,:,:,5:]

        # make cell offset martix
        h_indexs = tf.range(H, dtype=tf.float32)
        w_indexs = tf.range(W, dtype=tf.float32)
        x_cell_offsets, y_cell_offsets = tf.meshgrid(w_indexs, h_indexs)
        
        x_cell_offsets = tf.reshape(x_cell_offsets, [1,H,W,1])
        y_cell_offsets = tf.reshape(y_cell_offsets, [1,H,W,1])

        anchors = tf.constant(anchors, dtype=tf.float32, name="anchors")
        # print("anchors:\n", anchors)
        # # x_offsets = (x_cell_offsets + xy_offsets[:,:,:,:,0]) * 32 / img_width
        # x_offsets = (x_cell_offsets + xy_offsets[:,:,:,:,0]) / W
        # print(x_offsets)
        # # y_offsets = (y_cell_offsets + xy_offsets[:,:,:,:,1]) * 32 / img_height
        # y_offsets = (y_cell_offsets + xy_offsets[:,:,:,:,1]) / H
        # print(y_offsets)
        # w_scales = wh_scales[:,:,:,:,0] * anchors[:,0] / img_width
        # print(w_scales)
        # h_scales = wh_scales[:,:,:,:,1] * anchors[:,1] / img_height
        # print(h_scales)

        # Get boxes, x,y is normalized box center offset from left-upper corner by whole image scale
        # w,h is box size by whole image scale
        # boxes = tf.stack(
        #     [x_offsets, y_offsets, w_scales, h_scales],
        #     axis=-1)
        boxes = tf.stack([
            (x_cell_offsets + xy_offsets[:,:,:,:,0]) / W,
            (y_cell_offsets + xy_offsets[:,:,:,:,1]) / H,
            wh_scales[:,:,:,:,0] * anchors[:,0] / img_width,
            wh_scales[:,:,:,:,1] * anchors[:,1] / img_height],
            axis=-1)

        # print(boxes)
        # print(box_confidences)
        # print(class_probs)

        output = tf.reshape(
            tf.concat([
                boxes,
                tf.expand_dims(box_confidences, axis=-1),
                class_probs],
                axis=-1),
            [-1, H*W, B, 5+C],
            name='detection')

        # print("yolo output:\n", output)
    
    return output

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
