#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: wj
"""
import os
import tensorflow as tf
import modeling
import tokenization
import re

def get_labels(data_dir):
    labels_file=os.path.join(data_dir,'labels.txt')
    if 1:
        tf.logging.info("Read labels from '{}'".format(labels_file))
        labels=[]
        with open(labels_file,'r') as f:
            for line in f:
                line=line.strip()
                if line:
                    labels.append(line)
    return labels

def create_model(bert_config, input_ids, input_mask, segment_ids, num_labels,crf=False):
    model = modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=False)
    output_layer = model.get_sequence_output() #[batch_size, seq_length, hidden_size]
    hidden_size = output_layer.shape[-1].value
    #batch_size= output_layer.shape[0].value
    seq_length= output_layer.shape[1].value
    output_weights = tf.get_variable("output_weights", [num_labels, hidden_size],initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())
    with tf.variable_scope("loss"):
        output_layer=tf.reshape(output_layer,shape=[-1,hidden_size])
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        
        logits=tf.reshape(logits,shape=[-1,seq_length,num_labels])
        
        probabilities = tf.nn.softmax(logits, axis=-1)
        
        if crf:
            seq_length_list=tf.reduce_sum(input_mask,axis=-1)
            with tf.variable_scope("cft_loss"):
                transition_parameters=tf.get_variable("transitions",shape=[num_labels, num_labels],initializer=tf.contrib.layers.xavier_initializer())
            decode_tags,best_scores=tf.contrib.crf.crf_decode(logits,transition_parameters,seq_length_list)    
            return decode_tags
    return probabilities

class test_model:
    def __init__(self,out_dir='./out',crf=False):
        self.bert_config_file='./chinese_L-12_H-768_A-12/bert_config.json'
        self.batch_size=1
        self.max_length=64
        self.labels=get_labels('./data')
        self.vocab_file='./chinese_L-12_H-768_A-12/vocab.txt'
        self.bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)
        self.input_ids=tf.placeholder(name='input_ids',shape=[self.batch_size,self.max_length],dtype=tf.int32)
        self.input_mask=tf.placeholder(name='input_mask',shape=[self.batch_size,self.max_length],dtype=tf.int32)
        self.segment_ids=tf.placeholder(name='semgment_ids',shape=[self.batch_size,self.max_length],dtype=tf.int32)
        #labels
        if not crf:
            self.probs=create_model(self.bert_config,self.input_ids,self.input_mask,self.segment_ids,len(self.labels))
            self.pre_labels=tf.argmax(self.probs,axis=-1)
        else:
            self.decode_tags=create_model(self.bert_config,self.input_ids,self.input_mask,self.segment_ids,len(self.labels),True)
        self.restore_varibles=tf.trainable_variables()
        self.saver = tf.train.Saver(self.restore_varibles)
        self.model_save_file=tf.train.get_checkpoint_state(out_dir).model_checkpoint_path
        self.sess=tf.Session()
        self.saver.restore(self.sess,self.model_save_file)
        self.tokenizer=self.get_tokenizer(self.vocab_file)
        self.label_map=self.get_label_map(self.labels)
    def predict_crf(self,sentence):
        input_ids,input_mask,segment_ids,tokens_sentence=self.encode(sentence)
        feed_dict={self.input_ids:[input_ids],
                   self.input_mask:[input_mask],
                   self.segment_ids:[segment_ids]}
        decode_tags=self.sess.run(self.decode_tags,feed_dict)
        sentence,tokens_sentence,labels=self.decode(tokens_sentence,decode_tags)
        return sentence,tokens_sentence,labels
    def predict(self,sentence):
        input_ids,input_mask,segment_ids,tokens_sentence=self.encode(sentence)
        feed_dict={self.input_ids:[input_ids],
                   self.input_mask:[input_mask],
                   self.segment_ids:[segment_ids]}
        label_id,prob=self.sess.run([self.pre_labels,self.probs],feed_dict)
        sentence,tokens_sentence,labels=self.decode(tokens_sentence,label_id)
        return sentence,tokens_sentence,labels
        #label_id=label_id[0]
        #prob=prob[0][label_id]
        #return self.label_map[label_id],prob
    def decode(self,tokens_sentence,label_id):
        length=len(tokens_sentence)
        labels=[self.label_map[i] for i in label_id[0]][:length+1]
        sentence=''
        tag='o'
        num=0
        for token,label in zip(['']+tokens_sentence,labels):
            if num==0:
                sentence+=label
                num+=1
                continue
            if re.search("^##",token):
                sentence+=token
            elif re.search("^B_",label):
                if tag!='o':
                    sentence+=" </{}>".format(tag)
                tag=label[2:]
                sentence+=" <{}>".format(tag)
                sentence+=" "+token
            elif re.search("^M_",label):
                if tag==label[2:]:
                    sentence+=" "+token
                else:
                    if tag!='o':
                        sentence+=" </{}>".format(tag)
                    sentence+=" "+token
                    tag='o'
            else:
                if tag!='o':
                    sentence+=" </{}>".format(tag)
                sentence+=" "+token
                tag='o'
        if tag!='o':
           sentence+=" </{}>".format(tag)
           tag='o'
        return sentence,tokens_sentence,labels
            
            
        
    def get_label_map(self,label_list):
        label_map = {}
        for (i, label) in enumerate(label_list):
            label_map[i] = label
        return label_map
    def encode(self,sentence):
        tokens_sentence=self.tokenizer.tokenize(sentence)
        tokens=["[CLS]"]
        segment_ids=[0]
        for token in tokens_sentence:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == self.max_length
        assert len(input_mask) == self.max_length
        assert len(segment_ids) == self.max_length
        return input_ids,input_mask,segment_ids,tokens_sentence
    
    def get_tokenizer(self,vocab_file,do_lower_case=True):
        return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
