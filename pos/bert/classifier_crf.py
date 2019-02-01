#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: wj
"""
import tensorflow as tf
import glob
import os
import modeling
import tokenization
import optimization
import collections
import pickle
from sklearn.metrics import classification_report
import re

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids,segment_ids, label_ids,input_mask, output_mask):
      self.input_ids=input_ids
      self.segment_ids=segment_ids
      self.label_ids=label_ids
      self.input_mask=input_mask
      self.output_mask=output_mask

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
def get_labels(data_dir):
    labels_file=os.path.join(data_dir,'labels.txt')
    if os.path.isfile(labels_file):
        tf.logging.info("Read labels from '{}'".format(labels_file))
        labels=[]
        with open(labels_file,'r') as f:
            for line in f:
                line=line.strip()
                if line:
                    labels.append(line)
        return labels
    files=glob.glob(os.path.join(data_dir,'*_train.txt'))
    labels=set(['o','x','<PAD>'])
    for file in files:
        label=file.split("/")[-1].split('_')[0]
        labels.add(label)
        with open(file,'r') as f:
            for line in f:
                for slot_name in re.findall("<([a-zA-Z]+)>",line):
                    labels.add("B_"+slot_name)
                    labels.add("M_"+slot_name)
    labels=sorted(list(labels))
    with open(labels_file,'w') as f:
        for line in labels:
            f.write(line+'\n')
    return labels

def get_train_examples(raw_data_dir,data_mode):
    samples=[]
    num=0
    files=glob.glob(os.path.join(raw_data_dir,"*_"+data_mode+".txt"))
    for file in files:
        label=file.split("/")[-1].split('_')[0]
        with open(file,'r') as f:
            for line in f:
                if not line:
                    continue
                samples.append(InputExample(guid=num,text_a=line,label=label))
    return samples
def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,labels, output_mask,num_labels):
    model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
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
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer=tf.reshape(output_layer,shape=[-1,hidden_size])
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        
        logits=tf.reshape(logits,shape=[-1,seq_length,num_labels])
        seq_length_list=tf.reduce_sum(input_mask,axis=-1)
        
        with tf.variable_scope("cft_loss"):
            transition_parameters=tf.get_variable("transitions",shape=[num_labels, num_labels],initializer=tf.contrib.layers.xavier_initializer())
            log_likelihood, _= tf.contrib.crf.crf_log_likelihood(logits, labels,seq_length_list,transition_params=transition_parameters)
            loss = tf.reduce_mean(-log_likelihood,name="loss")
        
        decode_tags,best_scores=tf.contrib.crf.crf_decode(logits,transition_parameters,seq_length_list)
        
        pre_label=tf.cast(decode_tags,dtype=tf.int32)*tf.cast(output_mask,dtype=tf.int32)
        true_label=tf.cast(labels,dtype=tf.int32)*tf.cast(output_mask,dtype=tf.int32)
        sentence_level_acc=tf.reduce_sum(tf.cast(tf.equal(pre_label,true_label),dtype=tf.int32),axis=-1)
        #sentence_acc=tf.equal(accuracy,tf.constant([accuracy.shape[-1].value]*accuracy.shape[0].value))
        per_example_loss=-log_likelihood
        probabilities=tf.nn.softmax(logits,axis=-1)
    return (loss, per_example_loss, logits, probabilities,sentence_level_acc)
def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,sentence_acc_array=None):
    def model_fn(features, labels, mode):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        output_mask=features['output_mask']
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (total_loss, per_example_loss, logits, probabilities,sentence_level_acc) = create_model(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,output_mask,num_labels)
        batch_size=logits.shape[0].value
        seq_length=logits.shape[1].value
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.logging.info("**** Trainable Variables ****")
        init_num=0
        for var in tvars:
            #init_string = ""
            if var.name in initialized_variable_names:
                init_num+=1
                #init_string = ", *INIT_FROM_CKPT*"
                #tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,init_string)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("init from checkpoint done!var num:{}".format(init_num))
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps, False)
            output_spec=tf.estimator.EstimatorSpec(mode=mode,loss=total_loss,train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_ids, logits):
                print(sentence_level_acc)
                accuracy = tf.metrics.accuracy(sentence_level_acc,sentence_acc_array)
                loss = tf.metrics.mean(per_example_loss)
                return {
                        "eval_accuracy": accuracy,
                        "eval_loss": loss,
                        }
            eval_metrics = metric_fn(per_example_loss, label_ids, logits)
            output_spec = tf.estimator.EstimatorSpec(mode=mode,loss=total_loss,eval_metric_ops=eval_metrics)
        else:
            pre_label=tf.argmax(probabilities,axis=-1,name='label_prediction')#batch_size,seq_length,num_labels
            predictions={'pre_label':pre_label,'real_label':label_ids}
            output_spec = tf.estimator.EstimatorSpec(mode=mode,loss=total_loss,predictions=predictions)
        return output_spec
    return model_fn
def convert_single_example(ex_index, example, label_list, max_seq_length,tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    text_a=example.text_a
    labels_a=[]
    text_a=re.split("(<[a-zA-Z]+>[^<>]+</[a-zA-Z]+>)",text_a)
    tokens_a=[]
    for sub_text in text_a:
        if len(sub_text.strip())<1:
            continue
        elif re.search('<([a-zA-Z]+)>([^<>]+)<[/a-zA-Z]+>',sub_text):
            re_res=re.search('<([a-zA-Z]+)>([^<>]+)<[/a-zA-Z]+>',sub_text)
            slot_name=re_res.group(1)
            slot_value=re_res.group(2)
            slot_value=tokenizer.tokenize(slot_value)
            slot_labels=[]
            for i,s in enumerate(slot_value):
                if i==0:
                    slot_labels.append("B_"+slot_name)
                elif re.search("^##",s):
                    slot_labels.append("x")
                else:
                    slot_labels.append("M_"+slot_name)
            tokens_a.extend(slot_value)
            labels_a.extend(slot_labels)
        else:
            sub_text=tokenizer.tokenize(sub_text)
            sub_labels=['x' if re.search("^##",i) else 'o' for i in sub_text]
            tokens_a.extend(sub_text)
            labels_a.extend(sub_labels)
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    labels=[example.label]
    for label in labels_a:
        labels.append(label)
    labels.append('o')
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    output_mask=[1 if i!='x' else 0 for i in labels]
    label_ids=[label_map[i] for i in labels]
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        segment_ids.append(0)
        input_mask.append(0)
        output_mask.append(0)
        label_ids.append(label_map['<PAD>'])
    assert len(input_ids)==max_seq_length
    assert len(segment_ids)==max_seq_length
    assert len(label_ids)==max_seq_length
    assert len(input_mask)==max_seq_length
    assert len(output_mask)==max_seq_length
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(tokens))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("labels: %s" % " ".join([str(x) for x in labels]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        tf.logging.info("output_mask: %s" % " ".join([str(x) for x in output_mask]))
    feature = InputFeatures(
            input_ids=input_ids,
            segment_ids=segment_ids,
            label_ids=label_ids,
            input_mask=input_mask,
            output_mask=output_mask)
    return feature

def file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""
    writer = tf.python_io.TFRecordWriter(output_file)
    
    def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list,max_seq_length, tokenizer)
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["output_mask"] = create_int_feature(feature.output_mask)
        
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())

def file_based_input_fn_builder(input_file, seq_length, is_training,drop_remainder,batch_size,sample_length):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "output_mask": tf.FixedLenFeature([seq_length], tf.int64),
            }
    def input_fn():
        """The actual input function."""
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=sample_length)
        d = d.apply(
        tf.contrib.data.map_and_batch(
                lambda record: tf.parse_single_example(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
        return d
    return input_fn

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    output_dir='./out'
    data_dir='./data'
    bert_config_file='./chinese_L-12_H-768_A-12/bert_config.json'
    vocab_file='./chinese_L-12_H-768_A-12/vocab.txt'
    do_lower_case=True
    log_steps=200
    do_train=True
    do_eval=True
    do_predict=False
    train_batch_size=8
    eval_batch_size=32
    test_batch_size=32
    num_train_epochs=2
    warmup_proportion=0.1
    init_checkpoint='./chinese_L-12_H-768_A-12/bert_model.ckpt'
    eval_checkpoint='./out/model.ckpt-3668'
    learning_rate=5e-5
    max_seq_length=64
    output_predict_file='./predict.pkl'
    
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    tf.gfile.MakeDirs(output_dir)
    label_list=get_labels(data_dir)
    
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    run_config=tf.estimator.RunConfig(model_dir=output_dir,log_step_count_steps=log_steps,session_config=session_config)
    
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if do_train:
        train_examples = get_train_examples(data_dir,'train')
        num_train_steps = int(len(train_examples) / train_batch_size * num_train_epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)
    sentence_acc_array=[max_seq_length]*eval_batch_size
    model_fn=model_fn_builder(bert_config, len(label_list), init_checkpoint, learning_rate,num_train_steps, num_warmup_steps,sentence_acc_array=sentence_acc_array)
    estimator=tf.estimator.Estimator(model_fn=model_fn,model_dir=output_dir,config=run_config)
    if do_train:
        train_file = os.path.join(output_dir, "train.tf_record")
        file_based_convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d",train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
                input_file=train_file,
                seq_length=max_seq_length,
                is_training=True,
                drop_remainder=True,
                batch_size=train_batch_size,
                sample_length=len(train_examples))
        estimator.train(input_fn=train_input_fn, steps=num_train_steps)
    if do_eval:
        eval_examples = get_train_examples(data_dir,'test')
        num_eval_steps=int(len(eval_examples)/eval_batch_size)
        eval_file = os.path.join(output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
                eval_examples, label_list, max_seq_length, tokenizer, eval_file)
        tf.logging.info("***** Running eval *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d",eval_batch_size)
        tf.logging.info("  Num steps = %d", num_eval_steps)
        eval_input_fn=file_based_input_fn_builder(
                input_file=eval_file,
                seq_length=max_seq_length,
                is_training=False,
                drop_remainder=False,
                batch_size=eval_batch_size,
                sample_length=len(eval_examples))
        result=estimator.evaluate(input_fn=eval_input_fn,steps=num_eval_steps)#,checkpoint_path=eval_checkpoint)
        output_eval_file = os.path.join('./', "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    if do_predict:
        test_examples = get_train_examples(data_dir,'test')
        test_file = os.path.join(output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
                test_examples, label_list, max_seq_length, tokenizer, test_file)
        tf.logging.info("***** Running test *****")
        tf.logging.info("  Num examples = %d", len(test_examples))
        tf.logging.info("  Batch size = %d",test_batch_size)
        
        test_input_fn=file_based_input_fn_builder(
                input_file=test_file,
                seq_length=max_seq_length,
                is_training=False,
                drop_remainder=False,
                batch_size=test_batch_size)
        result=estimator.predict(test_input_fn)
        result=[i for i in result]
        true=[i['real_label'] for i in result]
        false=[i['pre_label'] for i in result]
        with open(output_predict_file,'wb') as f:
            pickle.dump(result,f,-1)
        with open("test_tmp.txt",'w') as f:
            res=classification_report(true,false)
            print(res)
            f.write(res)
        #with tf.gfile.GFile(output_predict_file, "w") as writer:
        #    tf.logging.info("***** Predict results *****")
        #    for prediction in result:
        #        output_line = "\t".join(str(class_probability) for class_probability in prediction) + "\n"
        #        writer.write(output_line)
        
        
if __name__=="__main__":
    #vocab_file='./chinese_L-12_H-768_A-12/vocab.txt'
    #do_lower_case=True
    #tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    #label_list=get_labels('./data')
    #example=InputExample(1,'hello newyork <artist>周杰</artist>，<song>king alen</song>小！',None,'music')
    #convert_single_example(1, example, label_list, 64,tokenizer)
    main()
