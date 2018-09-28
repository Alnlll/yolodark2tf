#!/usr/bin/env python
# -*- coding=utf-8 -*-

import numpy as np
import tensorflow as np

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
    S, S, B = box_confidences.shape
    S, S, C = class_probs.shape

    class_scores = np.zeros([S, S, B, C])
    for i in range(B):
        class_scores[:, :, i, :] = box_confidences[:, :, i][:,:,np.newaxis] * classes_probs
        print(class_scores[:, :, i, :].shape)
    
    print(np.max(class_scores))
    print(tf.keras.backend.max(class_scores, axis=-1))

    return class_scores