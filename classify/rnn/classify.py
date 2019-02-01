#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: wj
"""

import tensorflow as tf
import glob
import os
from tokenization import Tokenizer
#import basic_model_classify_01 as basic_model_classify
import basic_model_classify
import optimization
from sklearn.metrics import classification_report


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
              learning_rate,
              global_step,
          num_train_steps,
          end_learning_rate=0.0,
          power=1.0,
          cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    tvars = tf.trainable_variables()
    grads=tf.gradients(loss, tvars)
    grads, _ = tf.clip_by_global_norm(grads, 1.0)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op=optimizer.apply_gradients(zip(grads, tvars),global_step=global_step)
    
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op

class _log(tf.train.SessionRunHook):
    def begin(self):
        tf.logging.warn("Train begin...")
        self.steps=0
    def after_run(self, run_context, run_values):
        self.steps+=1
        if self.steps%100==0:
            tf.logging.warn("{}".format(run_values.results))
            tf.logging.warn("Steps: {}".format(self.steps))
class InputFeatures(object):
  """A single set of features of data."""
  def __init__(self, input_ids, input_mask, label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.label_id = label_id
class InputExample(object):
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label
def convert_single_example(index,example,max_seq_length,tokenizer):
    text=tokenizer.tokenize(example.text)
    if len(text)>max_seq_length:
        text=text[:max_seq_length]
    input_ids=tokenizer.convert_tokens_to_ids(text)
    input_mask = [1] * len(input_ids)
    label_id=tokenizer.convert_tag_to_ids(example.label)
    while len(input_ids)<max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
    feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            label_id=label_id)
    return feature

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
                samples.append(InputExample(num,line,label))
    return samples
def file_based_convert_examples_to_features(examples, save_dir, max_seq_length, tokenizer, output_file):
    vocab_file=os.path.join(save_dir,'vocab.txt')
    tag_file=os.path.join(save_dir,'tags.txt')
    tokenizer.load_vocab_tags(vocab_file,tag_file)
    if os.path.isfile(output_file):
        print("output file existed!")
        return
    writer = tf.python_io.TFRecordWriter(output_file)
    def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, max_seq_length, tokenizer)
        features = {}
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["label_ids"] = create_int_feature([feature.label_id])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
def create_vocab(raw_data_dir,save_dir,tokenizer):
    vocab_file=os.path.join(save_dir,'vocab.txt')
    tag_file=os.path.join(save_dir,'tags.txt')
    if os.path.isfile(vocab_file) and os.path.isfile(tag_file):
        print("vocab file existed!")
        return
    vocab=set()
    tags=set()
    files=glob.glob(os.path.join(raw_data_dir,'*.txt'))
    for file in files:
        tag=file.split("/")[-1].split("_")[0]
        tags.add(tag)
        with open(file,'r') as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                for token in tokenizer.tokenize(line):
                    vocab.add(token)
    vocab=['<pad>','<unk>']+list(vocab)
    with open(vocab_file,'w') as f:
        for line in vocab:
            f.write(line+'\n')
    with open(tag_file,'w') as f:
        for line in tags:
            f.write(line+'\n')
def create_model(config, is_training, input_ids, input_mask,labels,num_labels):
    if is_training:
        config.test=False
    else:
        config.test=True
    logits=basic_model_classify.bi_rnn_model(input_ids,input_mask,labels,config)
    with tf.variable_scope("loss"):
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    return (loss, per_example_loss, logits, probabilities)
def model_fn_builder(config, num_labels,  learning_rate,
                     num_train_steps, num_warmup_steps):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (total_loss, per_example_loss, logits, probabilities)=create_model(config,is_training,input_ids,input_mask,label_ids,num_labels)
        tvars = tf.trainable_variables()
        for var in tvars:
            tf.logging.info("  name = {}, shape = {}".format( var.name, var.shape))
        if mode == tf.estimator.ModeKeys.TRAIN:
            
            #train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps)
            train_op = create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps)
            output_spec= tf.estimator.EstimatorSpec(mode=mode,loss=total_loss,train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(per_example_loss, label_ids, logits):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(label_ids, predictions)
                loss = tf.metrics.mean(per_example_loss)
                return {
                        "eval_accuracy": accuracy,
                        "eval_loss": loss,
                        }
            eval_metrics = metric_fn(per_example_loss, label_ids, logits)
            output_spec = tf.estimator.EstimatorSpec(mode=mode,loss=total_loss,eval_metric_ops=eval_metrics)
        else:
            pre_label=tf.argmax(probabilities,axis=1,name='label_prediction')
            predictions={'pre_label':pre_label,'real_label':label_ids}
            output_spec = tf.estimator.EstimatorSpec(mode=mode,loss=total_loss,predictions=predictions)
        return output_spec
    return model_fn
def file_based_input_fn_builder(input_file, seq_length, is_training,drop_remainder):
    name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([], tf.int64) }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example
    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100000)

        d = d.apply(
                tf.contrib.data.map_and_batch(
                        lambda record: _decode_record(record, name_to_features),
                        batch_size=batch_size,
                        drop_remainder=drop_remainder))

        return d

    return input_fn

def train():
    tf.logging.set_verbosity(tf.logging.INFO)
    do_train=True
    do_eval=True
    do_test=True
    max_seq_length=50
    batch_size=256
    epochs=200
    warmup_proportion=0.1
    log_steps=500
    model_save_dir='./save'
    train_data_dir='./train_data'
    raw_data_dir='./data'
    tokenizer=Tokenizer()
    create_vocab(raw_data_dir,train_data_dir,tokenizer)
    train_examples = get_train_examples(raw_data_dir,'train')
    #print(len(train_examples))
    train_file = os.path.join(train_data_dir, "train.tf_record")
    file_based_convert_examples_to_features(train_examples, train_data_dir, max_seq_length, tokenizer, train_file)
    config=basic_model_classify.basic_config()
    config.vocab_size=len(tokenizer.vocab)
    config.max_length=max_seq_length
    config.n_tags=len(tokenizer.tags)
    config.batch_size=batch_size
    config.test=False
    num_train_steps=int(len(train_examples)/batch_size*epochs)
    num_warmup_steps=int(num_train_steps*warmup_proportion)
    #_trainining_hooks=_log()
    #_trainining_hooks=None
    model_fn=model_fn_builder(config=config, num_labels=config.n_tags, learning_rate=config.learning_rate,
                     num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps)
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    run_config=tf.estimator.RunConfig(model_dir=model_save_dir,log_step_count_steps=log_steps,session_config=session_config)
    estimater=tf.estimator.Estimator(model_fn=model_fn,model_dir=model_save_dir,params={'batch_size':config.batch_size},config=run_config)
    if do_train:
        tf.logging.info("train examples length:{}".format(len(train_examples)))
        tf.logging.info("train total steps:{}".format(num_train_steps))
        input_fn=file_based_input_fn_builder(train_file,config.max_length,True,True)
        estimater.train(input_fn,steps=num_train_steps)
    if do_eval:
        eval_examples=get_train_examples(raw_data_dir,'test')
        tf.logging.info("eval examples length:{}".format(len(eval_examples)))
        eval_file = os.path.join(train_data_dir, "test.tf_record")
        file_based_convert_examples_to_features(eval_examples, train_data_dir, max_seq_length, tokenizer, eval_file)
        input_fn=file_based_input_fn_builder(eval_file,config.max_length,False,False)
        num_eval_steps=int(len(eval_examples)/batch_size)
        tf.logging.info("eval total steps:{}".format(num_eval_steps))
        result=estimater.evaluate(input_fn,steps=num_eval_steps)
        output_eval_file = os.path.join('./', "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    if do_test:
        test_examples=get_train_examples(raw_data_dir,'test')
        tf.logging.info("test examples length:{}".format(len(test_examples)))
        test_file = os.path.join(train_data_dir, "test.tf_record")
        file_based_convert_examples_to_features(test_examples, train_data_dir, max_seq_length, tokenizer, test_file)
        input_fn=file_based_input_fn_builder(test_file,config.max_length,False,False)
        num_test_steps=int(len(test_examples)/batch_size)
        result=estimater.predict(input_fn)
        result=[i for i in result]
        true=[i['real_label'] for i in result]
        false=[i['pre_label'] for i in result]
        with open("test_tmp.txt",'w') as f:
            res=classification_report(true,false)
            print(res)
            f.write(res)
        #output_predict_file = os.path.join('./', "test_results.tsv")
        #with tf.gfile.GFile(output_predict_file, "w") as writer:
        #    tf.logging.info("***** Predict results *****")
        #    for prediction in result:
        #        output_line = "\t".join(str(class_probability) for class_probability in prediction) + "\n"
        #        writer.write(output_line)
        
if __name__=="__main__":
    train()
