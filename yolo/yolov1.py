#!/usr/bin/env python
#-*- coding=utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
try:
    import ConfigParser as configparser
except ImportError:
    import configparser
import math
import numpy as np
import tensorflow as tf

from utils.utils import make_unique_section_file
from utils.utils import get_padding_num

class DarkNet(object):
    def __init__(self, flags):
        self.cfg_parser = configparser.ConfigParser()
        self.sections = None
        self.params = {}
        self.configs = {}
        self.flags = flags

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    def __del__(self):
        del self.cfg_parser
        del self.sections
        del self.params
        del self.configs
        del self.flags
        self.sess.close()
    def parse_config(self, cfg_path):
        if not os.path.exists(cfg_path):
            raise IOError("{} doesn't exists.".format(cfg_path))
        # Get cfg file with unique section name
        uni_cfg_path, self.sections = make_unique_section_file(cfg_path)
        # Read cfg file
        self.cfg_parser.read(uni_cfg_path)

        # Get input image info
        try:
            self.img_height = self.cfg_parser.getint(self.sections[0], "height")
            self.img_width = self.cfg_parser.getint(self.sections[0], "width")
            self.img_channels = self.cfg_parser.getint(self.sections[0], "channels")
        except Exception as e:
            raise ValueError(e)
        
        # Get output feature map info
        try:
            self.classes = self.cfg_parser.getint(self.sections[-1], "classes")
            self.cell_size = self.cfg_parser.getint(self.sections[-1], "side")
            self.box_nums = self.cfg_parser.getint(self.sections[-1], "num")
        except Exception as e:
            raise ValueError(e)

        # print(self.img_height, self.img_width, self.img_channels)
        # print(self.classes, self.cell_size, self.box_nums)

        for section in self.sections[1:]:
            if section.startswith("convolutional"):
                activation = self.cfg_parser.get(section, 'activation')
                batch_norm, filters, size, stride, padding = map(
                    lambda x: self.cfg_parser.getint(section, x),
                    ['batch_normalize', 'filters', 'size', 'stride', 'pad']
                )
                self.configs[section] = {
                    'filters': filters, 'size': size, 'stride': stride, 'padding': padding,
                    'bn': batch_norm, 'activation': activation}

            if section.startswith("local"):
                size, stride, padding, filters = map(
                    lambda x: self.cfg_parser.getint(section, x),
                    ['size', 'stride', 'pad', 'filters']
                )
                activation = self.cfg_parser.get(section, 'activation')

                self.configs[section] = {
                    'filters': filters, 'size': size, 'stride': stride, 'padding': padding,
                    'activation': activation}

            if "pool" in section:
                size, stride = map(
                    lambda x: self.cfg_parser.getint(section, x),
                    ['size', 'stride'])
                if 'maxpool' in section:
                    pooling_type = 'max'
                else:
                    pooling_type = 'avg'
                
                self.configs[section] = {
                    'size': size, 'stride': stride, "type": pooling_type}

            if "dropout" in section:
                prob = self.cfg_parser.getfloat(section, 'probability')
                self.configs[section] = {
                    'prob': prob}

            if "connected" in section:
                n_out = self.cfg_parser.getint(section, 'output')
                activation = self.cfg_parser.get(section, 'activation')

                self.configs[section] = {
                    'out': n_out, 'activation': activation}

        # for section in self.sections[1:-1]:
        #     print(section, ":\n", self.configs[section])
    def activation(self, input, activation):
        if 'relu' == activation:
            output = tf.nn.relu(input, name = 'relu')
        elif 'leaky' == activation:
            output = tf.nn.relu(input, name = 'leaky')
        elif 'linear' == activation:
            output = input
        else:
            raise TypeError("Unknown activation type {}.".format(activation))
        
        return output
    def create_bn_layer(
        self,
        input,
        moving_decay=0.9,
        eps=1e-5,
        is_training=False,
        name='bn'):
        '''
        Create batch normalization layer
        '''
        with tf.variable_scope(name):
            # Data variance to learn
            gamma = tf.get_variable(
                "gamma", input.shape[-1],
                initializer=tf.constant_initializer(1.0), trainable=True)
            # Data mean to learn
            beta = tf.get_variable(
                "beta", input.shape[-1],
                initializer=tf.constant_initializer(0.0), trainable=True)

            # Calculate batch mean and variance
            axises = list(range((len(input.shape) - 1)))
            batch_mean, batch_var = tf.nn.moments(input, axises, name='moments')

            # Moving avergae for mean and variance
            ema = tf.train.ExponentialMovingAverage(decay=moving_decay)
            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                     return tf.identity(batch_mean), tf.identity(batch_var)
            
            # Update mean and var
            mean, var = tf.cond(
                tf.equal(is_training, True),
                mean_var_with_update,
                lambda: (ema.average(batch_mean), ema.average(batch_var))
            )
            output = tf.nn.batch_normalization(
                input, mean, var, beta, gamma, eps, name=name
            )

        return output, mean, var, beta, gamma
    def create_convolution_layer(
        self,
        input,
        filters, f_h, f_w,
        stride,
        padding=None,
        batch_norm=1,
        activation='leaky',
        is_training=False,
        filter_initializer=tf.variance_scaling_initializer(),
        use_bias=False,
        bias_initializer=tf.variance_scaling_initializer(),
        name = None):
        '''
        Create a convolution layer.
        Arguments:
        Returns:
        '''

        in_channels = input.shape[-1]
        # padding = 'SAME' if 1 == padding and 1 == stride else 'VALID'
        padding = 'SAME' if 1== padding else 'VALID'

        with tf.variable_scope(name):
            # Get filter weight
            filter = tf.get_variable(
                'kernel', 
                shape=[f_h, f_w, in_channels, filters],
                initializer=filter_initializer)
            self.params[name] = {'kernel': filter}
            print("\n",filter)
            # Convolution
            output = tf.nn.conv2d(
                input,
                filter,
                [1, stride, stride, 1],
                padding,
                name='conv')
            if use_bias:
                bias = tf.get_variable(
                    'bias',
                    shape = [filters],
                    initializer=bias_initializer
                )
                self.params[name] = {'bias': bias}
                output = tf.add(output, bias)

            if batch_norm:
                output, mean, var, beta, gamma = self.create_bn_layer(
                    output,
                    moving_decay=0.9,
                    eps=1e-5,
                    is_training=False,
                    name='bn'
                )
                self.params[name]['bn'] = {
                    'mean': mean, 'var': var, 'beta': beta, 'gamma': gamma}

            # activation
            if activation:
                output = self.activation(output, activation)

        return output
    def create_local_convolution_layer(
        self,
        input,
        filters, f_h, f_w,
        stride,
        padding=0,
        activation='linear',
        filter_initializer=tf.variance_scaling_initializer(),
        name = None):
        ''' 
        Locally-connected layer
        '''

        if 4 != len(input.shape):
            raise ValueError("Only support 4 dimension input, recieved {} input {}".format(
                len(input.shape), input))
        # padding, impleme = ('same',2) if 1 == padding and 1 == stride else ('valid',1)
        padding = 'same' if 1 == padding else 'valid'

        with tf.variable_scope(name):
            # Pre-padding because local layer doesn't support 'same' padding.
            if 'same' == padding:
                paddings = get_padding_num(input.shape, [1, f_h, f_w, 1], stride)
                paddings[0, :], paddings[-1, :] = 0, 0
                input = tf.pad(input, tf.constant(paddings), name='pad')

            output = tf.keras.layers.LocallyConnected2D(
                filters=filters,
                kernel_size=(f_h, f_w),
                input_shape=input.shape[-3:],
                strides=(stride, stride),
                padding='valid',
                activation=None,
                kernel_initializer=filter_initializer)(input)

            kernel = self.sess.graph.get_tensor_by_name("{}/locally_connected2d/kernel:0".format(name))
            bias = self.sess.graph.get_tensor_by_name("{}/locally_connected2d/bias:0".format(name))
            self.params[name] = {'kernel': kernel, 'bias': bias}

            if activation:
                output = self.activation(output, activation)

        return output
    def create_pooling_layer(
        self,
        input,
        p_h, p_w,
        pooling_type,
        stride,
        padding="VALID",
        name=None):
        '''
        Pooling layer
        '''
        with tf.name_scope(name):
            if 'avg' == pooling_type:
                output = tf.nn.avg_pool(
                    input,
                    (1, p_h, p_w, 1),
                    (1, stride, stride, 1),
                    padding,
                    name='avg_pooling')
            elif 'max' == pooling_type:
                output = tf.nn.max_pool(
                    input,
                    (1, p_h, p_w, 1),
                    (1, stride, stride, 1),
                    padding,
                    name='avg_pooling')
            else:
                raise TypeError("Required 'avg' or 'max', but received '{}'.".format(pooling_type))

        return output
    def create_dropout_layer(
        self,
        input,
        prob,
        is_training=False,
        seed=None,
        name=None):
        '''
        Dropout Layer.
        '''
        with tf.name_scope(name):
            input = tf.layers.flatten(input, name='flatten')

            if is_training:
                output = tf.nn.dropout(input, prob, seed=seed, name='dropout')
            else:
                output = input

        return output
    def create_fully_connectd_layer(
        self,
        input,
        n_out,
        activation='leaky',
        weight_initializer=tf.variance_scaling_initializer(),
        use_bias=True,
        bias_initializer=tf.variance_scaling_initializer(),
        name=None):
        '''
        Fully-Connected layer.
        '''

        with tf.variable_scope(name):
            # Do flatten
            input = tf.layers.flatten(input, name='flatten')
            # Get weight
            n_in = input.shape[-1]
            W = tf.get_variable(
                'weight', 
                [n_in, n_out],
                initializer=weight_initializer,
                dtype=tf.float32)
            # Get bias
            if use_bias:
                b = tf.get_variable(
                    'bias',
                    [n_out],
                    initializer=bias_initializer,
                    dtype=tf.float32)
                output = tf.nn.bias_add(tf.matmul(input, W), b)
            else:
                output = tf.matmul(input, W)

            if activation:
                output = self.activation(output, activation)

        return output
    def create_model(self):
        # Parse model config
        self.parse_config(self.flags.cfg)

        # Create input placeholder
        input = tf.placeholder(
            tf.float32,
            shape=[None, self.img_height, self.img_width, self.img_channels],
            name="input")
        
        output = input

        # Create model by config file
        for index, section in enumerate(self.sections[1:]):
            if section.startswith("convolutional"):
                output = self.create_convolution_layer(
                    output,
                    self.configs[section]['filters'],
                    self.configs[section]['size'],
                    self.configs[section]['size'],
                    self.configs[section]['stride'],
                    self.configs[section]['padding'],
                    bool(self.configs[section]['bn']),
                    self.configs[section]['activation'],
                    use_bias=not self.configs[section]['bn'],
                    is_training=self.flags.train,
                    name=section)
                print("output shape: {}".format(output.shape))

            if section.startswith("local"):
                output = self.create_local_convolution_layer(
                    output,
                    self.configs[section]['filters'],
                    self.configs[section]['size'],
                    self.configs[section]['size'],
                    self.configs[section]['stride'],
                    self.configs[section]['padding'],
                    self.configs[section]['activation'],
                    name=section)
                print("output shape: {}".format(output.shape))

            if "pool" in section:
                output = self.create_pooling_layer(
                    output,
                    self.configs[section]['size'],
                    self.configs[section]['size'],
                    self.configs[section]['type'],
                    self.configs[section]['stride'],
                    padding='VALID',
                    name=section)
                print("output shape: {}".format(output.shape))

            if "dropout" in section:
                output = self.create_dropout_layer(
                    output,
                    self.configs[section]['prob'],
                    is_training=self.flags.train,
                    seed=None,
                    name=section)
                print("output shape: {}".format(output.shape))

            if "connected" in section:
                output = self.create_fully_connectd_layer(
                    output,
                    self.configs[section]['out'],
                    activation=self.configs[section]['activation'],
                    name=section)
                print("output shape: {}".format(output.shape))
            print()
    def load_weight(self):
        if not os.path.exists(self.flags.weight):
            raise IOError("{} doesn't exist.".format(self.flags.weight))
        
        with open(self.flags.weight, 'rb') as f:
            self.header = np.fromfile(f, dtype=np.int32, count=5)
            self.seen = self.header[3]

            print(self.header)
            print(self.seen)

            weights = np.fromfile(f, dtype=np.float32)
