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
# from tensorflow.python import debug as tf_debug

from utils.utils import make_unique_section_file
from utils.utils import get_padding_num
from utils.utils import print_conv_layer_params
from utils.utils import print_pooling_layer_params
from utils.utils import print_dropout_layer_params
from utils.utils import print_upsample_layer_params
from utils.utils import print_route_params
from utils.utils import print_fc_layer_params
from utils.utils import print_v3_detection_layer_params
from utils.utils import timer_wrapper
from yolo_utils import *
# from yolo_utils import select_boxes_by_classes_prob
# from yolo_utils import non_max_suppression
# from yolo_utils import read_classes_names


_LEAKY_RATIO = .1
_BATCH_NORM_DECAY = .9
_BATCH_NORM_EPSILON = 1e-05
_WEIGHT_HEADER_LEN = 5

import sys

class Yolov3(object):
    def __init__(self, flags):
        self.cfg_parser = configparser.ConfigParser()
        self.sections = None
        self.params = {}
        self.configs = {}
        # layer important values: input, output
        self.vals = {}
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

        # Post process encoding
        scores, boxes, classes = self._build_detector(self.encoding)
        # print(scores)
        # print(boxes)
        # print(classes)

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
        if self.verbose:
            print("\nParsing config file {}...\n-----------------------------------------".format(cfg_path))
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
            
            # yolov3 doesn't use fc layer, different size of input can be feeded.
            self.img_size = self.flags.img_size
            self.img_channels = self.cfg_parser.getint(self.sections[0], "channels")
        except Exception as e:
            raise ValueError(e)
        
        # Get output feature map info
        # try:
        #     self.classes = self.cfg_parser.getint(self.sections[-1], "classes")
        #     self.box_nums = self.cfg_parser.getint(self.sections[-1], "num")
        #     self.use_softmax = self.cfg_parser.getboolean(self.sections[-1], 'softmax')
        # except Exception as e:
        #     raise ValueError(e)

        for index, section in enumerate(self.sections[1:]):
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
            
            if section.startswith('route'):
                from_layer = self.cfg_parser.get(section, 'layers')
                if ',' in from_layer:
                    from_layer = from_layer.strip().split(', ')
                    # from_layer = [int(val) for val in from_layer]
                    # from_layer = map(lambda x: x if x>0 else (index+x))
                    from_layer = [int(val) if int(val)>0 else (index+int(val)) for val in from_layer]
                else:
                    from_layer = int(from_layer) if int(from_layer) > 0 else (int(from_layer)+index)
                self.configs[section] = {'from_layer': from_layer}

            if section.startswith('upsample'):
                stride = self.cfg_parser.getint(section, 'stride')
                self.configs[section] = {'stride': stride}
            
            if section.startswith('yolo'):
                classes = self.cfg_parser.getint(section, 'classes')
                box_nums = self.cfg_parser.getint(section, 'num')
                
                masks = self.cfg_parser.get(section, 'mask').strip().split(',')
                masks = [int(mask) for mask in masks]

                anchors = self.cfg_parser.get(self.sections[-1], 'anchors').strip().split(",  ")
                anchors = [[float(val.split(',')[0]), float(val.split(',')[1])] for val in anchors]

                self.classes = classes

                self.configs[section] = {
                    'classes': classes, 'box_nums': len(masks),
                    'mask': masks, 'anchors': anchors}

            if self.verbose:
                print("{} - {}: {}".format(index, section, self.configs[section]))
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
        self.vals[name] = {"input": inputs, "output": output}

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

        self.vals[name] = {"input": inputs, "output": output}

        if self.verbose:
            print_pooling_layer_params(
                [p_h, p_w], stride, padding, pooling_type,
                inputs.shape.as_list(), output.shape.as_list(), name=name)
        return output
    def _create_flatten_layer(self, inputs, transpose=[0, 3, 1, 2]):
        inputs = tf.transpose(inputs, transpose)
        output = create_flatten_layer(inputs, name="flatten")
        
        self.vals[name] = {"input": inputs, "output": output}

        return output
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
        self.vals[name] = {"input": inputs, "output": output}

        if self.verbose:
            print_fc_layer_params(
                param['weight_shape'], activation,
                inputs.shape.as_list(), output.shape.as_list(), name=name)

        return output
    def _create_upsample_layer(self, inputs, config, name=None):
        try:
            stride = config['stride']
        except Exception as e:
            raise Exception(e)
        
        output = create_upsample_layer(inputs, stride, name=name)
        
        self.vals[name] = {'inputs': inputs, 'output': output}

        if self.verbose:
            print_upsample_layer_params(
                stride,
                inputs.shape.as_list(), output.shape.as_list(),
                name=name)

        return output
    def _create_route_layer(self, config, name=None):
        routes = config['from_layer']
        get_layer_name = lambda x: self.sections[1:][x]

        if isinstance(routes, int):
            layer_names = get_layer_name(routes)
            output = self.vals[layer_names]['output']
        elif isinstance(routes, list):
            layer_names = [get_layer_name(layer) for layer in routes]
            output = reduce(
                lambda x,y: tf.concat(
                    [self.vals[x]['output'], self.vals[y]['output']], axis=-1),
                layer_names)
        else:
            raise TypeError("Need from_layer of type int or list, got {}".format(type(routes)))
        
        if self.verbose:
            print_route_params(
                routes, layer_names,
                output.shape.as_list(), name=name)

        return output
    def _create_detection_layer(self, inputs, config, name=None):
        # Get anchors by mask
        mask = config['mask']
        anchors = config['anchors']
        anchors = [m for i,m in enumerate(anchors) if i in mask]
        classes = config['classes']

        output = create_v3_detection_layer(inputs, classes, anchors, name=name)
        self.vals[name] = {'inputs': inputs, 'output': output}

        if self.verbose:
            print_v3_detection_layer_params(
                inputs.shape.as_list(), output.shape.as_list(),
                name=name)

        return output
    def _gather_detections(self, name='gather_detections'):
        detection_list = []
        with tf.name_scope(name):
            for section in self.sections[1:]:
                if section.startswith('yolo'):
                    detection_list.append(self.vals[section]['output'])
            if detection_list:
                if 1 == len(detection_list):
                    detections = detection_list[0]
                else:
                    detections = reduce(
                        lambda det1, det2: tf.concat([det1, det2], axis=1),
                        detection_list)
            else:
                detections = None

        return detections
    def _load_model(self, inputs):
        if self.verbose:
            print("\nConstructing network...\n-----------------------------------------")
        output = inputs

        # Exclude net layer for route index align
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
            if section.startswith("dropout"):
                output = self._create_dropout_layer(
                    output,
                    self.configs[section]['prob'],
                    is_training=self.flags.train,
                    seed=None,
                    name=section)
                # print("output shape: {}".format(output.shape))
            if section.startswith("connected"):
                output = self._create_fully_connectd_layer(
                    output,
                    self.configs[section]['out'],
                    activation=self.configs[section]['activation'],
                    name=section)
                # print("output shape: {}".format(output.shape))
            if section.startswith("upsample"):
                output = self._create_upsample_layer(output, self.configs[section], name=section)
            if section.startswith('route'):
                output = self._create_route_layer(self.configs[section], name=section)
            if section.startswith('yolo'):
                _ = self._create_detection_layer(output, self.configs[section], name=section)
        # Gather output from different scale branch
        gathered = self._gather_detections()
        if gathered is not None:
            output = gathered

        return output
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
    def _load_weight(self, header_len=4):
        if not os.path.exists(self.flags.weight):
            raise IOError("{} doesn't exist.".format(self.flags.weight))

        if self.verbose:
            print("\nLoading weights...\n-----------------------------------------")

        with open(self.flags.weight, 'rb') as f:
            # different from v1,v2, v3 header's length is 5
            self.header = np.fromfile(f, dtype=np.int32, count=header_len)
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
                    print("layer:{} load {} float data  total:{}/{}".format(
                        format(section, '<20s'),
                        format(ptr-prev_ptr, '<10d'),
                        ptr+header_len,
                        weights.shape[-1]+header_len))
            self.sess.run(assign_op_list)
    def _load_model_weight(self):
        if self.flags.weight:
            self._load_weight(_WEIGHT_HEADER_LEN)
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
    def _pre_process_input(self, inputs):
        with tf.name_scope('pre_process'):
            image = tf.to_float(inputs) / 255.0
            image = tf.image.resize_images(
                image, tf.constant([self.img_size, self.img_size], dtype=tf.int32))
        return image
    def _build_network(self):
        # Input placeholder
        inputs = tf.placeholder(
            tf.float32,
            shape=[None, None, None, self.img_channels],
            name="inputs")

        # pre-processed image data
        processed_input = self._pre_process_input(inputs)

        # Builde network
        output = self._load_model(processed_input)
        self._load_model_weight()

        return inputs, output, processed_input
    def _build_detector(self, encoding):
        with tf.name_scope("post_process"):
            boxes, box_confidences, class_probs = tf.split(
                encoding, [4, 1, self.classes],
                axis=-1,
                name='detections')

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
