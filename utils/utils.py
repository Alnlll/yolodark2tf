#!/usr/bin/env python
# -*- coding=utf-8 -*-

import os
from io import StringIO
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