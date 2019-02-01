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
        self.num_layers=1
        self.grad_clip=1
        self.learning_rate=0.02
def bi_rnn_model(input_ids,input_mask,label_id,config):
    seq_length=tf.reduce_sum(input_mask,axis=-1)
    embedding = tf.get_variable("embedding", [config.vocab_size, config.hidden_size], dtype=tf.float32)
    x_train_embedding=tf.nn.embedding_lookup(embedding,input_ids)
    layer_input=x_train_embedding
    for i in range(config.num_layers):
        if not config.test:
            layer_input=tf.nn.dropout(layer_input,config.keep_prob)
        lstm_cell=tf.nn.rnn_cell.LSTMCell(config.hidden_size)
        if i == 0:
            # don't add skip connection from token embedding to
            # 1st layer output
            pass
        else:
            # add a skip connection
            lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)
        variable_scope_name = 'RNN_{0}/RNN/MultiRNNCell/Cell{1}'.format('forward', i)
        with tf.variable_scope(variable_scope_name):
            layer_output, final_state = tf.nn.dynamic_rnn(
                    lstm_cell,
                    layer_input,
                    sequence_length=seq_length,
                    dtype=tf.float32
                    )
        layer_input = layer_output
    layer_input=tf.reduce_sum(layer_input,axis=1)
    logits=tf.layers.dense(layer_input,config.n_tags)
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
     
        
        
        
