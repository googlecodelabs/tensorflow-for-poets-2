#!/usr/bin/python
#
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import timeit
import sys

import numpy as np
import PIL.Image as Image

import tensorflow as tf

            
def main(image_path, graph_file="tf_files/retrained_graph.pb"):
    # Read in the image_data
    
    image = Image.open(image_path)
    image = np.array(image, dtype=np.float32)
    

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile("tf_files/retrained_labels.txt")]
    
    # Unpersists graph from file
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()

        with tf.gfile.FastGFile(graph_file, 'rb') as f:
            graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, name='')

    with tf.Session(graph=graph) as sess:        
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = graph.get_tensor_by_name('final_result:0')

        start_time = timeit.default_timer()
        predictions = sess.run(softmax_tensor, 
                 {'Cast:0': image})
        elapsed = timeit.default_timer() - start_time
        print('Elapsed time: %f\n' % elapsed)    
        
        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
        
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main(*sys.argv[1:])

