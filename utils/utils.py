#!/usr/bin/env python
# -*- coding=utf-8 -*-

from __future__ import division

import os
import time
from io import StringIO

import numpy as np
import tensorflow as tf

def load_pb(path):
    if not os.path.exists(path):
        raise IOError(path)
    
    with tf.gfile.FastGFile(path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read()) 
        # Imports the graph from graph_def into the current default Graph. 
        tf.import_graph_def(graph_def, name='')
def make_unique_section_file(path):
    if not os.path.exists(path):
        raise IOError("{} not exists.".format(path))
    
    path_prefix = os.path.split(path)[0]
    fname = os.path.split(path)[1].split(".")[0]
    uni_file_path = os.path.join(path_prefix, "{}_uni.cfg".format(fname))
    
    section_counts = {}
    uni_section_names = []
    with open(path, "r+") as f:
        with StringIO() as uni_sec_cfg:
            for line in f.readlines():
                # Get new section
                if line.startswith("["):
                    section_name = line.strip('[]\n')
                    # Count same section
                    if not section_name in section_counts:
                        section_counts[section_name] = 1
                    else:
                        section_counts[section_name] += 1
                    uni_sec_name = "{}_{}".format(section_name, section_counts[section_name])
                    uni_section_names.append(uni_sec_name)
                    uni_sec_name = "[{}]\n".format(uni_sec_name)
                    uni_sec_cfg.write(uni_sec_name.decode())
                else:
                    uni_sec_cfg.write(line.decode())
            with open(uni_file_path, 'w') as uni_f:
                uni_f.write(uni_sec_cfg.getvalue())

    return uni_file_path, uni_section_names
def get_padding_num(input_shape, kernel_size, stride):
    '''
    Get number of 0 to be padded.
    '''
    paddings = np.zeros([len(input_shape), 2])
    # paddings_fix_odd = np.zeros([len(input_shape), 2])
    input_shape = np.array(input_shape)
    kernel_size = np.array(kernel_size)

    # print(input_shape.shape)
    # print(kernel_size.shape)
    # print(paddings.shape)

    paddings[:, 0] = np.ceil((kernel_size - 1) / stride / 2)
    paddings[:, 1] = np.ceil((kernel_size - 1) / stride / 2)
    
    # Seem like darknet same padding doesn't fix odd dimension here.
    # padded_shape = input_shape + 2*paddings[:, 0]
    # mask = (0 != padded_shape % 2)
    # paddings[:, 1] += (np.ones(input_shape.shape) * mask)

    # print(paddings)
    # print(paddings_fix_odd)

    return paddings.astype(int)
def print_conv_layer_params(
    kernel_size, stride, padding, bn, activation, input_shape, output_shape, name="Convlayer"):
    f_h, f_w, _, filters = kernel_size
    print("layer:{} size:{}x{} / {} filters:{} padding:{} bn:{} activation:{} shape:{}->{}".format(
        format(name, '<20s'),
        format(f_h, 'd'),
        format(f_w, 'd'),
        format(stride, '<5d'),
        format(filters, '<10d'),
        format(padding, '<10s'),
        format(bn, "<4b"),
        format(activation, '<10s'),
        input_shape,
        output_shape
    ))
def print_pooling_layer_params(
    pooling_size, stride, padding, pooling_type, input_shape, output_shape, name="PoolingLayer"):
    p_h, p_w = pooling_size
    print("layer:{} size:{}x{} / {} padding:{} type:{} shape:{}->{}".format(
        format(name, '<20s'),
        format(p_h, 'd'),
        format(p_w, 'd'),
        format(stride, '<5d'),
        format(padding, '<10s'),
        format(pooling_type, "<10s"),
        input_shape,
        output_shape
    ))
def print_dropout_layer_params(
    prob, input_shape, output_shape, name="DropoutLayer"):
    print("layer:{} prob:{} shape:{}->{}".format(
        format(name, '<20s'),
        format(prob, '<10.4f'),
        input_shape,
        output_shape
    ))
def print_fc_layer_params(
    weight_shape, activation, input_shape, output_shape, name="FCLayer"):
    print("layer:{} weight:{}x{} activation:{} shape:{}->{}".format(
        format(name, '<20s'),
        format(weight_shape[0], 'd'),
        format(weight_shape[1], '<5d'),
        format(activation, '<10s'),
        input_shape,
        output_shape
    ))
def print_upsample_layer_params(
    factor, input_shape, output_shape, name="Route"):
    print("layer:{} factor:x{} shape:{}->{}".format(
        format(name, '<20s'),
        format(factor, '<8d'),
        input_shape,
        output_shape
    ))
def print_route_params(
    routes, route_names, output_shape, name="Route"):
    print("layer:{} routes:{} route_names:{}    shape:->{}".format(
        format(name, '<20s'),
        format(str(routes), '<9s'),
        route_names,
        output_shape
    ))
def print_shortcut_params(
    from_layer, activation, input_shape, output_shape, name="Shortcut"):
    print("layer:{} from:{} activation:{} shape:{}->{}".format(
        format(name, '<20s'),
        format(from_layer, '<11d'),
        format(activation, '<7s'),
        input_shape,
        output_shape
    ))
def print_v3_detection_layer_params(
    input_shape, output_shape, name="yolo"):
    print("layer:{} shape:{}->{}".format(
        format(name, '<20s'),
        input_shape,
        output_shape
    ))
def timer_wrapper(func):
    def wrapper(*args, **args2):
        t0 = time.time()
        print "@%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__)
        wrapped = func(*args, **args2)
        print "@%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__)
        print "@%.3fs taken for {%s}" % (time.time() - t0, func.__name__)
        return wrapped
    return wrapper
