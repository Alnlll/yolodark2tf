#!/usr/bin/env python
#-*- coding=utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import math
import random
import colorsys
try:
    import ConfigParser as configparser
except ImportError:
    import configparser
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from utils.utils import make_unique_section_file
from utils.utils import get_padding_num
from utils.utils import print_conv_layer_params
from utils.utils import print_pooling_layer_params
from utils.utils import print_dropout_layer_params
from utils.utils import print_fc_layer_params
from utils.utils import timer_wrapper
from yolo_utils import *
# from yolo_utils import select_boxes_by_classes_prob
# from yolo_utils import non_max_suppression
# from yolo_utils import read_classes_names


_LEAKY_RATIO = .1
_BATCH_NORM_DECAY = .9
_BATCH_NORM_EPSILON = 1e-05

import sys

class Yolov2(object):
    def __init__(self, flags):
        self.cfg_parser = configparser.ConfigParser()
        self.sections = None
        self.params = {}
        self.configs = {}
        self.verbose = True
        self.flags = flags

        self.classes_names = read_classes_names(self.flags.names)

        # gpu_options = tf.GPUOptions(allow_growth=True)
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = tf.Session()

        # Parse cfg file
        self._parse_config(self.flags.cfg)

        # Construct network and load weight
        self.input, self.encoding, self.processed_image= self._build_network()
        
        # whole network processing equal downsampling with ratio 32
        # self.output_size = self.img_size // 32

        # Post process encoding
        scores, boxes, classes = self._build_detector(self.encoding)

        # Define detect function
        self.detect_func = lambda image: self.sess.run(
            [scores, boxes, classes], feed_dict={self.input: image})

        # Get summary writer
        if self.flags.summary:
            writer = tf.summary.FileWriter(self.flags.summary, self.sess.graph)
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
            # self.img_height = self.cfg_parser.getint(self.sections[0], "height")
            # self.img_width = self.cfg_parser.getint(self.sections[0], "width")
            
            # yolov2 doesn't use fc layer, different size of input can be feeded.
            self.img_size = self.flags.img_size
            self.img_channels = self.cfg_parser.getint(self.sections[0], "channels")
        except Exception as e:
            raise ValueError(e)
        
        # Get output feature map info
        try:
            self.classes = self.cfg_parser.getint(self.sections[-1], "classes")
            self.box_nums = self.cfg_parser.getint(self.sections[-1], "num")
            self.use_softmax = self.cfg_parser.getboolean(self.sections[-1], 'softmax')

            anchors = self.cfg_parser.get(self.sections[-1], 'anchors').strip().split(",  ")
            self.anchors = [[float(val.split(',')[0]), float(val.split(',')[1])] for val in anchors]
        except Exception as e:
            raise ValueError(e)

        for section in self.sections[1:]:
            if section.startswith("convolutional"):
                activation = self.cfg_parser.get(section, 'activation')
                filters, size, stride, padding = map(
                    lambda x: self.cfg_parser.getint(section, x),
                    ['filters', 'size', 'stride', 'pad']
                )
                try:
                    batch_norm = self.cfg_parser.getint(section, 'batch_normalize')
                except:
                    batch_norm = 0
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
    def _activation(self, inputs, activation):
        return activation_func(inputs, activation, leaky_ratio=_LEAKY_RATIO)
    def _create_bn_layer(
        self,
        inputs,
        momentum=0.9,
        eps=1e-5,
        is_training=False,
        name=None):
        '''
        Create batch normalization layer
        '''
        return create_bn_layer(
            inputs,
            momentum=momentum,
            eps=eps,
            is_training=is_training,
            name=name)
    def _create_convolution_layer(
        self,
        inputs,
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

        output, param = create_convolution_layer(
            inputs,
            filters, f_h, f_w,
            stride,
            padding=padding,
            batch_norm=batch_norm,
            momentum=_BATCH_NORM_DECAY,
            eps=_BATCH_NORM_EPSILON,
            activation=activation,
            is_training=is_training,
            filter_initializer=filter_initializer,
            use_bias=use_bias,
            bias_initializer=bias_initializer,
            name=name
        )
        self.params[name] = param

        if self.verbose:
            padding = 'SAME' if 1 == padding and 1 == stride else 'VALID'
            print_conv_layer_params(
                self.params[name]['weight_shape'], stride, padding, batch_norm, activation,
                inputs.shape.as_list(), output.shape.as_list(), name=name)
        return output
    def _create_pooling_layer(
        self,
        inputs,
        p_h, p_w,
        pooling_type,
        stride,
        padding="VALID",
        name=None):
        '''
        Pooling layer
        '''

        padding = 'SAME' if None == padding and 1 == stride else 'VALID'

        output = create_pooling_layer(
            inputs,
            p_h, p_w,
            pooling_type,
            stride,
            padding=padding,
            name=name)

        if self.verbose:
            print_pooling_layer_params(
                [p_h, p_w], stride, padding, pooling_type,
                inputs.shape.as_list(), output.shape.as_list(), name=name)
        return output
    def _create_flatten_layer(self, inputs, transpose=[0, 3, 1, 2]):
        inputs = tf.transpose(inputs, transpose)
        return create_flatten_layer(inputs, name="flatten")
    def _create_fully_connectd_layer(
        self,
        inputs,
        n_out,
        activation='leaky',
        weight_initializer=tf.variance_scaling_initializer(),
        use_bias=True,
        bias_initializer=tf.variance_scaling_initializer(),
        name=None):
        '''
        Fully-Connected layer.
        '''
        inputs = self._create_flatten_layer(inputs)
        output, param = create_fully_connectd_layer(
            inputs,
            n_out,
            activation=activation,
            activation_fn=None,
            weight_initializer=weight_initializer,
            use_bias=use_bias,
            bias_initializer=bias_initializer,
            name=name)
        self.params[name] = param

        if self.verbose:
            print_fc_layer_params(
                param['weight_shape'], activation,
                inputs.shape.as_list(), output.shape.as_list(), name=name)

        return output
    def _load_model(self, input):
        if self.verbose:
            print("\nConstructing network...\n-----------------------------------------")
        
        output = input

        # Create model by config file
        for section in self.sections[1:]:
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
                    padding=None,
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

        return output
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

                kernel = tf.get_variable("conv2d/kernel")
                # DaekNet weight shape [f, c, h, w], tensorflow [h, w, c, f]
                kernel_shape = self.params[section]['weight_shape']
                kernel_data = weights[ptr:ptr + num].reshape(
                    kernel_shape[3], kernel_shape[2], kernel_shape[0], kernel_shape[1])
                kernel_data = np.transpose(kernel_data, [2, 3, 1, 0])
                assign_ops.append(kernel.assign(kernel_data))

                # self.sess.run(assign_ops[-5:])
                # print("beta:\n", self.sess.run(beta))
                # print("gamma:\n", self.sess.run(gamma))
                # print("mean:\n", self.sess.run(mean))
                # print("var: \n", self.sess.run(var))
                # print("kernel:\n", self.sess.run(kernel))

                ptr += num
            else:
                bias = tf.get_variable("conv2d/bias")
                assign_ops.append(bias.assign(weights[ptr:ptr + filters]))
                ptr += filters

                kernel = tf.get_variable("conv2d/kernel")
                kernel_shape = self.params[section]['weight_shape']
                kernel_data = weights[ptr:ptr + num].reshape(
                    kernel_shape[3], kernel_shape[2], kernel_shape[0], kernel_shape[1])
                kernel_data = np.transpose(kernel_data, [2, 3, 1, 0])
                assign_ops.append(kernel.assign(kernel_data))
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

        self.sess.run(assign_ops)
        return ptr, assign_ops
    def _load_fc_weight(self, section, config, ptr, weights):
        n_in, n_out = self.params[section]['weight_shape']

        assign_ops = []
        with tf.variable_scope(section, reuse=True):
            bias = tf.get_variable("fully_connected/biases")
            assign_ops.append(bias.assign(weights[ptr:ptr + n_out]))
            ptr += n_out

            weight = tf.get_variable("fully_connected/weights")
            weight_data = weights[ptr:ptr + n_out*n_in].reshape([n_out, n_in])
            weight_data = np.transpose(weight_data, [1, 0])

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

        if self.verbose:
            print("\nLoading weights...\n-----------------------------------------")

        with open(self.flags.weight, 'rb') as f:
            self.header = np.fromfile(f, dtype=np.int32, count=4)
            self.seen = self.header[3]

            weights = np.fromfile(f, dtype=np.float32)
            ptr = 0
            assign_op_list = []

            # load weight doesn't run out of weight file, have no idea about remaining data.
            for section in self.sections:
                prev_ptr = ptr
                if section.startswith("convolutional"):
                    config = self.configs[section]
                    ptr, assign_ops = self._load_conv_weight(section, config, ptr, weights)
                    assign_op_list.insert(-1, assign_ops)

                if section.startswith("local"):
                    config = self.configs[section]
                    ptr, assign_ops = self._load_local_conv_weight(section, config, ptr, weights)
                    assign_op_list.insert(-1, assign_ops)

                if section.startswith("connected"):
                    config = self.configs[section]
                    ptr, assign_ops = self._load_fc_weight(section, config, ptr, weights)
                    assign_op_list.insert(-1, assign_ops)

                if self.verbose:
                    # weights.shape[-1]-ptr
                    print("layer:{} load {} float data  total:{}/{}".format(
                        format(section, '<20s'),
                        format(ptr-prev_ptr, '<10d'),
                        # ptr-prev_ptr,
                        ptr+4,
                        weights.shape[-1]+4))
            self.sess.run(assign_op_list)
    def _load_model_weight(self):
        if self.flags.weight:
            self._load_weight()
        elif self.flags.ckpt:
            self._load_ckpt()
        elif self.flags.pb:
            self._load_pb()
        else:
            raise TypeError("Please provide model weight file.")
    @timer_wrapper
    def _inference(self, image):
        if not isinstance(image, np.ndarray):
            raise TypeError("Need np.ndarray, got {} of type {}.".format(image, type(image)))
        return self.detect_func(image[np.newaxis])
    def _pre_process_input(self, input):
        image = tf.to_float(input) / 255.0
        image = tf.image.resize_images(
            image, tf.constant([self.img_size, self.img_size], dtype=tf.int32))
        return image
    def _build_network(self):
        # Input placeholder
        input = tf.placeholder(
            tf.float32,
            shape=[None, None, None, self.img_channels],
            name="input")

        # Handler image data
        processed_input = self._pre_process_input(input)

        # Builde network
        output = self._load_model(processed_input)
        self._load_model_weight()

        return input, output, processed_input
    def _build_detector(self, encoding):
        with tf.name_scope("post_process"):
            _, S, S, _ = encoding.shape.as_list()
            B, C = self.box_nums, self.classes

            detect_outputs = tf.reshape(encoding, [-1, S, S, B, C + 5])

            # bx = sigmoid(tx) + cx, by = sigmoid(ty) + cy, shape:[None, S*S, B, 2]
            xy_offsets = tf.nn.sigmoid(detect_outputs[:,:,:,:,0:2], name='xy')
            # bw = pw * exp(tw), bh = ph * exp(th), shape:[None, S*S, B, 2]
            wh_scales = tf.exp(detect_outputs[:,:,:,:,2:4], name='wh')
            # confidence = sigmoid(to), shape:[None, S*S, B]
            box_confidences = tf.nn.sigmoid(detect_outputs[:,:,:,:,4], name='box_confidences')
            # transform independent logistic prob results to softmax results, shape[None, S*S, B, classes]
            if self.use_softmax:
                class_probs = tf.nn.softmax(detect_outputs[:,:,:,:,5:], name="class_probs")
            else:
                class_probs = detect_outputs[:,:,:,:,5:]

            # print(xy_offsets)
            # print(wh_scales)
            # print(box_confidences)
            # print(class_probs)

            # make cell offset martix
            h_indexs = tf.range(S, dtype=tf.float32)
            w_indexs = tf.range(S, dtype=tf.float32)
            x_cell_offsets, y_cell_offsets = tf.meshgrid(w_indexs, h_indexs)
            # print(x_cell_offsets)
            # print(y_cell_offsets)
            x_cell_offsets = tf.reshape(x_cell_offsets, [1,S,S,1])
            y_cell_offsets = tf.reshape(y_cell_offsets, [1,S,S,1])

            anchors = tf.constant(self.anchors, dtype=tf.float32, name="anchors")
            # print(anchors)

            # Get boxes, x,y is normalized box center offset from left-upper corner by whole image scale
            # w,h is box size by whole image scale
            boxes = tf.stack([
                (x_cell_offsets + xy_offsets[:,:,:,:,0]) / S,
                (y_cell_offsets + xy_offsets[:,:,:,:,1]) / S,
                wh_scales[:,:,:,:,0] * anchors[:,0] / S,
                wh_scales[:,:,:,:,1] * anchors[:,1] / S],
                axis=4)

            # select those boxes whose class confidence is larger than threshold
            class_scores, boxes, box_classes = select_boxes_by_classes_prob(
                box_confidences, class_probs, boxes,
                threshold=self.flags.score_thresh)
            # Do nms, get final results
            scores, boxes, classes = non_max_suppression(
                class_scores, boxes, box_classes,
                iou_threshold=self.flags.iou_thresh)

            return scores, boxes, classes
    def show_results(
            self,
            image,
            scores, boxes, classes,
            show=True,
            text_record=None,
            output_file="detected.jpg"):
        """Show the detection boxes

        """

        # Get color map
        hsv_tuples = [(x/float(len(self.classes_names)), 1., 1.)  for x in range(len(self.classes_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
        random.seed(23333)
        # random.seed(9527)
        random.shuffle(colors)
        random.seed(None)

        img_cp = image.copy()
        if text_record:
            if not os.path.exists(os.path.split(text_record)[0]):
                os.makedirs(os.path.split(text_record)[0])
            f = open(text_record, "w")

        detect_count = scores.shape[0]
        for i in range(detect_count):
            box = boxes[i]
            class_name = self.classes_names[classes[i]]
            score = scores[i]

            x, y, w, h = box.astype(np.int32)

            result_string = "Class name: {} [x, y, w, h]=[{}, {}, {}, {}], score={}".format(
                format(class_name, '<10s'),
                format(x, '<3d'),
                format(y, '<3d'),
                format(w, '<3d'),
                format(h, '<3d'),
                format(score, '<5.4f'))

            if self.verbose:
                print(result_string)
            if text_record:
                f.write("{}\n".format(result_string))
            if show or output_file:
                # cv2.rectangle(img_cp, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 255, 0), 2)
                cv2.rectangle(img_cp, (x - w//2, y - h//2), (x + w//2, y + h//2), colors[classes[i]], 2)

                cv2.rectangle(img_cp, (x - w//2, y - h//2 + 20), (x + w//2, y - h//2), colors[classes[i]], -1)
                cv2.putText(img_cp, 
                            "{}:{}".format(class_name, format(score, '.4f')), 
                            (x - w//2 + 5, y - h//2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                if show:
                    cv2.imwrite(output_file, img_cp)
                if output_file:
                    cv2.imwrite(output_file, img_cp)
        if text_record:
            f.close()
    def detect_from_image_file(self, image_file):
        image = cv2.imread(image_file)
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # print("input imageRGB:", image_RGB)

        # print("input processed image:\n", self.sess.run(self.processed_image, feed_dict={self.input: image_RGB[np.newaxis]}))
        # print("encoding:\n", self.sess.run(self.encoding, feed_dict={self.input: image_RGB[np.newaxis]}))
        # print("conv1 input:\n", self.sess.run("resize_images/ResizeBilinear:0", feed_dict={self.input: image_RGB[np.newaxis]}))
        # print("conv1 output:\n", self.sess.run("convolutional_1/leaky:0", feed_dict={self.input: image_RGB[np.newaxis]}))
        # print("pool1 output:\n", self.sess.run("maxpool_1/avg_pooling:0", feed_dict={self.input: image_RGB[np.newaxis]}))
        # print("conv8 output:\n", self.sess.run("convolutional_8/leaky:0", feed_dict={self.input: image_RGB[np.newaxis]}))
        # print("flatten output:\n", self.sess.run("connected_1/flatten/Reshape:0", feed_dict={self.input: image_RGB[np.newaxis]}))
        
        # "convolutional_1/leaky:0"
        # "convolutional_2/leaky:0"
        # "convolutional_3/leaky:0"
        # "convolutional_4/leaky:0"
        # "convolutional_5/leaky:0"
        # "convolutional_6/leaky:0"
        # "convolutional_7/leaky:0"
        # "convolutional_8/leaky:0"
        # "maxpool_1/avg_pooling:0"
        # "maxpool_2/avg_pooling:0"
        # "maxpool_3/avg_pooling:0"
        # "maxpool_4/avg_pooling:0"
        # "maxpool_5/avg_pooling:0"
        # "maxpool_6/avg_pooling:0"
        # "connected_1/flatten/Reshape:0"

        if self.verbose:
            print("\nDetecting {}...\n-----------------------------------------".format(image_file))
            print("Detected result:")
        scores, boxes, classes = self._inference(image_RGB)

        img_h, img_w, _ = image.shape
        boxes[:,0] *= (1.0 * img_w)
        boxes[:,1] *= (1.0 * img_h)
        boxes[:,2] *= (1.0 * img_w)
        boxes[:,3] *= (1.0 * img_h)

        if self.flags.text_record:
            if self.flags.output_dir:
                text_record = os.path.join(self.flags.output_dir, self.flags.text_record)
            else:
                text_record = self.flags.text_record
        else:
            text_record = None

        if self.flags.output_dir:
            output_file = os.path.join(self.flags.output_dir, "detected.jpg")
        self.show_results(
            image,
            scores, boxes, classes,
            show=True,
            text_record=text_record,
            output_file=output_file)
