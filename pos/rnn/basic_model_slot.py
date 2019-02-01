#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: wj
"""

import tensorflow as tf


class basic_config:
    def __init__(self):
        self.vocab_size=''
        self.max_length=''
        self.n_tags=''
        self.batch_size=''
        self.cell_type='lstm'
        self.hidden_size=256
        self.test=False
        self.keep_prob=0.9
        self.num_layers=3
        self.grad_clip=1
        self.learning_rate=0.02

def bi_rnn_model(input_ids,input_mask,label_ids,is_training,config):
    embedding = tf.get_variable("embedding", [config.vocab_size, config.hidden_size], dtype=tf.float32)
    x_train_embedding=tf.nn.embedding_lookup(embedding,input_ids)
    seq_lengths=tf.reduce_sum(input_mask,axis=-1)
    def basic_cell():
        return tf.nn.rnn_cell.LSTMCell(config.hidden_size)
    def bi_rnn_layer(fw_cell, bw_cell, inputs, seq_lengths, scope=None):
        (outputs_fw,outputs_bw),_=tf.nn.bidirectional_dynamic_rnn(fw_cell,bw_cell,inputs=inputs,\
                                                                        dtype=tf.float32,\
                                                                        sequence_length=seq_lengths,\
                                                                        scope=scope)
            #inputs:if time_major == False (default), this must be a tensor of shape:`[batch_size, max_time, input_size]`.
        outputs=tf.concat([outputs_fw, outputs_bw], axis=-1)
            #outputs = outputs_fw + outputs_bw
        return outputs#shape=[batch_size,max_time,2*hidden_size]
    def multi_bi_rnn_layer(inputs, seq_lengths, num_layers, is_test, keep_prob):
        inner_outputs = inputs
        for n in range(num_layers):
            forward_cell = basic_cell()
            backward_cell =basic_cell()
            inner_outputs = bi_rnn_layer(forward_cell, backward_cell, inner_outputs, seq_lengths, 'brnn_{}'.format(n))
            if not is_test:
                inner_outputs = tf.contrib.layers.dropout(inner_outputs, keep_prob=keep_prob, is_training=True)
        return inner_outputs#shape=[batch_size, max_time, input_size]`
    outputs=multi_bi_rnn_layer(x_train_embedding,seq_lengths,config.num_layers, not is_training,config.keep_prob)
    outputs=tf.reshape(outputs,shape=(-1,2*config.hidden_size))
    return outputs
if __name__=="__main__":
    config=basic_config()     
    config.vocab_size=1000
    config.num_layers=3
    config.test=False
    config.keep_prob=0.9
    import numpy as np
    input_ids=np.ones([10,64],dtype=np.int32)
    input_mask=np.ones([10,64],dtype=np.int32)
    label_ids=np.ones([10,64],dtype=np.int32)
    model=bi_rnn_model(input_ids,input_mask,label_ids,config)
        
        
        
