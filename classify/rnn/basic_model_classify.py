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
def bi_rnn_model(input_ids,input_mask,label_id,config):
    seq_length=tf.reduce_sum(input_mask,axis=-1)
    embedding = tf.get_variable("embedding", [config.vocab_size, config.hidden_size], dtype=tf.float32)
    x_train_embedding=tf.nn.embedding_lookup(embedding,input_ids)
    def basic_cell():
        if config.cell_type=='lstm':
            return tf.nn.rnn_cell.LSTMCell(config.hidden_size)
        elif config.cell_type=='gru':
            return tf.nn.rnn_cell.GRUCell(config.hidden_size)
    def bi_rnn_layer(fw_cell, bw_cell, inputs, seq_lengths, scope=None):
        (outputs_fw,outputs_bw),_=tf.nn.bidirectional_dynamic_rnn(fw_cell,\
                                                                        bw_cell,inputs=inputs,\
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
    with tf.variable_scope('rlan') as scope:
            outputs=multi_bi_rnn_layer(x_train_embedding,seq_length,config.num_layers,config.test,config.keep_prob)
            #outputs=tf.reshape(outputs,shape=(-1,2*self.hidden_size))
            outputs=tf.reduce_sum(outputs,axis=1)#shape=[batch_size,2*hidden_size]
            #print(outputs)
    W_out = tf.get_variable("W_out",shape = [2*config.hidden_size, config.n_tags],initializer=tf.contrib.layers.xavier_initializer())
    b_out = tf.Variable(tf.constant(0.0, shape=[config.n_tags]), name="b_out")
        
    logits = tf.nn.xw_plus_b(outputs, W_out, b_out, name="logits") #[batch_size  , n_tags]
    return logits

if __name__=="__main__":
    config=basic_config()
    config.vocab_size=1000
    config.max_length=20
    config.n_tags=3
    config.batch_size=5
    import numpy as np
    input_ids=np.ones((5,20),dtype=np.int32)
    input_mask=np.ones((5,20),dtype=np.int32)
    label_id=[1,1,1,1,1]
    l=bi_rnn_model(input_ids,input_ids,label_id,config)
     
        
        
        
