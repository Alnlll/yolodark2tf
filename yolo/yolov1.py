#!/usr/bin/env python
#-*- coding=utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import math
try:
    import ConfigParser as configparser
except ImportError:
    import configparser
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from utils.utils import make_unique_section_file
from utils.utils import get_padding_num
from utils.utils import print_conv_layer_params
from utils.utils import print_pooling_layer_params
from utils.utils import print_dropout_layer_params
from utils.utils import print_fc_layer_params
from yolo_utils import select_boxes_by_classes_prob

class DarkNet(object):
    def __init__(self, flags):
        self.cfg_parser = configparser.ConfigParser()
        self.sections = None
        self.params = {}
        self.configs = {}
        self.flags = flags

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        # Construct network and load weight
        self.input, self.output = self._load_model()
        self._load_model_weight()
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
    def __del__(self):
        del self.cfg_parser
        del self.sections
        del self.params
        del self.configs
        del self.flags
        self.sess.close()
    def _parse_config(self, cfg_path):
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
    def _activation(self, input, activation):
        if 'relu' == activation:
            output = tf.nn.relu(input, name = 'relu')
        elif 'leaky' == activation:
            output = tf.nn.relu(input, name = 'leaky')
        elif 'linear' == activation:
            output = input
        else:
            raise TypeError("Unknown activation type {}.".format(activation))
        
        return output
    def _create_bn_layer(
        self,
        input,
        # moving_decay=0.9,
        # eps=1e-5,
        is_training=False,
        name='bn'):
        '''
        Create batch normalization layer
        '''
        # with tf.variable_scope(name):
        output = tf.layers.batch_normalization(
            input,
            training=is_training
            # name='bn'
        )

        # gamma, beta, mean, var = 
        # gamma = self.sess.get_tensor_by_name("{}/batch_normalization/gamma:0".format())
        
        # return output
            # # Data variance to learn
            # gamma = tf.get_variable(
            #     "gamma", input.shape[-1],
            #     initializer=tf.constant_initializer(1.0), trainable=True)
            # # Data mean to learn
            # beta = tf.get_variable(
            #     "beta", input.shape[-1],
            #     initializer=tf.constant_initializer(0.0), trainable=True)

            # # Calculate batch mean and variance
            # axises = list(range((len(input.shape) - 1)))
            # batch_mean, batch_var = tf.nn.moments(input, axises, name='moments')

            # # Moving avergae for mean and variance
            # ema = tf.train.ExponentialMovingAverage(decay=moving_decay)
            # def mean_var_with_update():
            #     ema_apply_op = ema.apply([batch_mean, batch_var])
            #     with tf.control_dependencies([ema_apply_op]):
            #          return tf.identity(batch_mean), tf.identity(batch_var)
            
            # # Update mean and var
            # mean, var = tf.cond(
            #     tf.equal(is_training, True),
            #     mean_var_with_update,
            #     lambda: (ema.average(batch_mean), ema.average(batch_var))
            # )
            # print(mean, var)
            # output = tf.nn.batch_normalization(
            #     input, mean, var, beta, gamma, eps, name=name
            # )

        return output
    def _create_convolution_layer(
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
            self.params[name] = {'kernel': filter, 'weight_shape': [f_h, f_w, in_channels, filters]}
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
                self.params[name]['bias'] = bias
                output = tf.add(output, bias)

            if batch_norm:
                output = tf.layers.batch_normalization(
                    output,
                    training=is_training
                    # name='bn'
                )

            # activation
            if activation:
                output = self._activation(output, activation)

            self.params[name]['output_shape'] = output.shape
        print_conv_layer_params(
            filter.shape.as_list(), stride, padding, batch_norm, activation,
            input.shape.as_list(), output.shape.as_list(), name=name)

        return output
    def _create_local_convolution_layer(
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
                input = tf.pad(input, tf.constant(paddings, dtype=tf.int32), name='pad')

            # output = tf.keras.layers.LocallyConnected2D(
            #     filters=filters,
            #     kernel_size=(f_h, f_w),
            #     input_shape=input.shape[-3:],
            #     strides=(stride, stride),
            #     padding='valid',
            #     activation=None,
            #     kernel_initializer=filter_initializer)(input)
            local_conv = tf.keras.layers.LocallyConnected2D(
                filters=filters,
                kernel_size=(f_h, f_w),
                input_shape=input.shape[-3:],
                strides=(stride, stride),
                padding='valid',
                activation=None,
                kernel_initializer=filter_initializer)
            output = local_conv(input)
            local_weights = local_conv.get_weights()
            print(local_weights)
            print(type(local_weights))
            print(len(local_weights))
            print(local_weights[0].shape)
            print(local_weights[1].shape)

            kernel = self.sess.graph.get_tensor_by_name("{}/locally_connected2d/kernel:0".format(name))
            print(kernel)
            print(kernel.shape)
            # bias = self.sess.graph.get_tensor_by_name("{}/locally_connected2d/bias:0".format(name))
            self.params[name] = {
                'weight_shape': (f_h, f_w, input.shape.as_list()[-1], filters),
                'output_shape': output.shape}

            if activation:
                output = self._activation(output, activation)

        # print(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        print("input:\n", input, input.dtype)
        print("output:\n", output, output.dtype)
        return output
    def _create_pooling_layer(
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
        print_pooling_layer_params(
            [p_h, p_w], stride, padding, pooling_type,
            input.shape.as_list(), output.shape.as_list(), name=name)
        return output
    def _create_dropout_layer(
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
        print_dropout_layer_params(
            prob, input.shape.as_list(), output.shape.as_list(), name=name)

        return output
    def _create_fully_connectd_layer(
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
            self.params[name] = {'weight': W, 'weight_shape': [n_in, n_out]}
            # Get bias
            if use_bias:
                b = tf.get_variable(
                    'bias',
                    [n_out],
                    initializer=bias_initializer,
                    dtype=tf.float32)
                self.params[name]['bias'] = b
                output = tf.nn.bias_add(tf.matmul(input, W), b)
            else:
                output = tf.matmul(input, W)

            if activation:
                output = self._activation(output, activation)

            self.params[name]['output_shape'] = output.shape
        print_fc_layer_params(
            W.shape.as_list(), activation,
            input.shape.as_list(), output.shape.as_list(), name=name)

        return output
    def _load_model(self):
        # Parse model config
        self._parse_config(self.flags.cfg)

        # Create input placeholder
        input = tf.placeholder(
            tf.float32,
            shape=[None, self.img_height, self.img_width, self.img_channels],
            name="input")
        
        output = input

        # Create model by config file
        for index, section in enumerate(self.sections[1:]):
            if section.startswith("convolutional"):
                output = self._create_convolution_layer(
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
                # print("output shape: {}".format(output.shape))

            if section.startswith("local"):
                output = self._create_local_convolution_layer(
                    output,
                    self.configs[section]['filters'],
                    self.configs[section]['size'],
                    self.configs[section]['size'],
                    self.configs[section]['stride'],
                    self.configs[section]['padding'],
                    self.configs[section]['activation'],
                    name=section)
                # print("output shape: {}".format(output.shape))

            if "pool" in section:
                output = self._create_pooling_layer(
                    output,
                    self.configs[section]['size'],
                    self.configs[section]['size'],
                    self.configs[section]['type'],
                    self.configs[section]['stride'],
                    padding='VALID',
                    name=section)
                # print("output shape: {}".format(output.shape))

            if "dropout" in section:
                output = self._create_dropout_layer(
                    output,
                    self.configs[section]['prob'],
                    is_training=self.flags.train,
                    seed=None,
                    name=section)
                # print("output shape: {}".format(output.shape))

            if "connected" in section:
                output = self._create_fully_connectd_layer(
                    output,
                    self.configs[section]['out'],
                    activation=self.configs[section]['activation'],
                    name=section)
                # print("output shape: {}".format(output.shape))
        self.sess.run(tf.global_variables_initializer())

        return input, output
        # print("output:\n",output)
    def _load_conv_weight(self, section, config, ptr, weights):
        # shape of kernel
        f_h, f_w, prev_c, filters = self.params[section]['weight_shape']
        # kernel variable size
        num = f_h * f_w * prev_c * filters

        assign_ops = []
        with tf.variable_scope(section, reuse=True):
            if config['bn']:
                beta = tf.get_variable("batch_normalization/beta")
                # assign_ops.append(tf.assign(beta, weights[ptr:ptr + filters]))
                assign_ops.append(beta.assign(weights[ptr:ptr+filters]))
                ptr += filters

                gamma = tf.get_variable("batch_normalization/gamma")
                assign_ops.append(gamma.assign(weights[ptr:ptr + filters]))
                ptr += filters

                mean = tf.get_variable("batch_normalization/moving_mean")
                assign_ops.append(mean.assign(weights[ptr:ptr + filters]))
                ptr += filters

                var = tf.get_variable("batch_normalization/moving_variance")
                assign_ops.append(var.assign(weights[ptr:ptr + filters]))
                ptr += filters

                kernel = tf.get_variable("kernel")
                # DaekNet weight shape [f, c, h, w], tensorflow [h, w, c, f]
                kernel_shape = self.params[section]['kernel'].shape
                kernel_data = weights[ptr:ptr + num].reshape(
                    kernel_shape[3], kernel_shape[2], kernel_shape[0], kernel_shape[1])
                kernel_data = np.transpose(kernel_data, [2, 3, 1, 0])
                assign_ops.append(kernel.assign(kernel_data))
                ptr += num
            else:
                bias = tf.get_variable("bias")
                assign_ops.append(bias.assign(weights[ptr:ptr + filters]))
                ptr += filters

                kernel = tf.get_variable("kernel")
                assign_ops.append(kernel.assign(weights[ptr:ptr + num]))
                ptr += num
        
        return ptr, assign_ops
    def _load_local_conv_weight(self, section, config, ptr, weights):
        f_h, f_w, prev_c, filters = self.params[section]['weight_shape']
        _, o_h, o_w, o_c = self.params[section]['output_shape']

        # local conv locations
        n_locs = o_h * o_w
        # kernel size
        size = f_h * f_w * prev_c * filters * n_locs

        assign_ops = []
        with tf.variable_scope(section, reuse=True):
            # bias = tf.get_variable("locally_connected2d/bias")
            n_bias = o_h * o_w * o_c
            bias = tf.Variable("locally_connected2d/bias")
            assign_ops.append(bias.assign(weights[ptr:ptr + n_bias].tostring()))
            ptr += n_bias
            
            kernel_data = weights[ptr:ptr + size].reshape(
                [o_h*o_w, filters, prev_c, f_h, f_w])
            kernel_data = np.transpose(kernel_data, [0, 3, 4, 2, 1])
            
            # weight_data = [kernel_data.tostring(), np.zeros([o_h, o_w, filters]).tostring()]
            kernel = tf.Variable("locally_connected2d/kernel")
            assign_ops.append(kernel.assign(kernel_data.tostring()))
            ptr += size

            # print('load local {} {} {}'.format(n_bias, size, size+n_bias))
            # print("load local conv n_bias:{} size:{}  loaded:{}".format(n_bias, size, size+bias))
        
        self.sess.run(assign_ops)
        return ptr, assign_ops
    def _load_fc_weight(self, section, config, ptr, weights):
        n_in, n_out = self.params[section]['weight_shape']

        print(n_in, n_out)
        assign_ops = []
        with tf.variable_scope(section, reuse=True):
            bias = tf.get_variable("bias")
            assign_ops.append(bias.assign(weights[ptr:ptr + n_out]))
            ptr += n_out

            weight = tf.get_variable("weight")
            weight_data = weights[ptr:ptr + n_out*n_in].reshape([n_in, n_out])
            # print(n_out + n_out*n_in)
            # print(weight_data.shape)
            # print(weights.shape[-1])
            # print(ptr + n_out * n_in)

            assign_ops.append(weight.assign(weight_data))
            ptr += n_out * n_in

        self.sess.run(assign_ops)
        return ptr, assign_ops
    def _load_ckpt(self):
        pass
    def _load_pb(self):
        pass
    def _load_weight(self):
        if not os.path.exists(self.flags.weight):
            raise IOError("{} doesn't exist.".format(self.flags.weight))
        
        with open(self.flags.weight, 'rb') as f:
            self.header = np.fromfile(f, dtype=np.int32, count=4)
            self.seen = self.header[3]

            weights = np.fromfile(f, dtype=np.float32)
            ptr = 0
            assign_op_list = []

            # load weight doesn't run out of weight file, have no idea about remaining data.
            for index,section in enumerate(self.sections):
                prev_ptr = ptr
                if section.startswith("convolutional"):
                    config = self.configs[section]
                    ptr, assign_ops = self.load_conv_weight(section, config, ptr, weights)
                    assign_op_list.insert(-1, assign_ops)

                if section.startswith("local"):
                    config = self.configs[section]
                    ptr, assign_ops = self.load_local_conv_weight(section, config, ptr, weights)
                    assign_op_list.insert(-1, assign_ops)
                    
                if section.startswith("connected"):
                    config = self.configs[section]
                    ptr, assign_ops = self.load_fc_weight(section, config, ptr, weights)
                    assign_op_list.insert(-1, assign_ops)

                print("{} layer {} load {} float data\n".format(index, section, ptr-prev_ptr))
                print("{}/{} {}".format(ptr+4, weights.shape[-1]+4, weights.shape[-1]-ptr))
            self.sess.run(assign_op_list)
            print("load {} float data\n".format(ptr))
    def _load_model_weight(self):
        if self.flags.weight:
            self._load_weight()
        elif self.flags.ckpt:
            self._load_ckpt()
        elif self.flags.pb:
            self._load_pb()
        else:
            raise TypeError("Please provide model weight file.")
    def _inference(self, image):
        input = self.sess.graph.get_tensor_by_name("input:0")
        output = self.sess.graph.get_tensor_by_name("connected_1/BiasAdd:0")

        # print(self.sess.run("local_1/pad:0", feed_dict={input: image}))
        # return None

        res = self.sess.run(output, feed_dict={input: image})
        return res
    def _pre_process_input(self, input):
        image = tf.to_float(input) / 255.0
        image = tf.image.resize_images(
            image, tf.constant([self.img_height, self.img_width], dtype=tf.int32))
        return image
    def _build_detector(self, encoding):
        with tf.name_scope("post_process"):
            S, B, C = self.cell_size, self.box_nums, self.classes
            
            idx1 = S * S * C
            idx2 = idx1 + S*S*B
            class_prob = tf.reshape(encoding[:, :idx1], [S, S, 1, C])
            box_confidences = tf.reshape(encoding[:, idx1:idx2, B, 1])
            boxes = tf.reshape(encoding[:, idx2:], [S, S, B, 4])

            # Restore boxes values
            # x,y is offset from cell's left-up corner within 0-1
            # w,h is sqrt result of scale by image height and width
            x_offset = np.transpose(
                np.reshape(
                    np.array(
                        [np.arange(S)] * S * B),
                        [B, S, S]),
                    [1, 2, 0])
            y_offset = np.transpose(x_offset, [1, 0, 2])

            # now all coordinate is offset from left-up corner within 0-1
            boxes = tf.stack([
                (boxes[:, :, :, 0] + tf.constant(x_offset, dtype=tf.float32)) / S,
                (boxes[:, :, :, 1] + tf.constant(y_offset, dtype=tf.float32)) / S,
                tf.square(boxes[:, :, :, 2]),
                tf.square(boxes[:, :, :, 3])],
                axis=3)

            


    def detect_from_image_file(self, image_file):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = self.inference(image)
        
        # for i in range(res.shape[-1]):
        #     print(res[0, i], " ", end="")
        #     if 19 == i % 20:
        #         print("\n")
        S, B, C = self.cell_size, self.box_nums, self.classes
        class_probs = res[:, :S*S*C].reshape(S,S,1,C)
        box_confidences = res[:, S*S*C:S*S*C+S*S*B].reshape(S,S,B,1)
        boxes = res[:, -S*S*4:].reshape(S,S,4)
        print(class_probs.shape)
        print(box_confidences.shape)
        print(boxes.shape)
        scores = select_boxes_by_classes_prob(box_confidences, class_probs)
        print(scores)
        print(np.max(scores))

