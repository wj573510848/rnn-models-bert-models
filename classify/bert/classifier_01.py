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

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
def get_labels(data_dir):
    files=glob.glob(os.path.join(data_dir,'*_train.txt'))
    labels=set()
    for file in files:
        label=file.split("/")[-1].split('_')[0]
        labels.add(label)
    return sorted(list(labels))

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
def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,labels, num_labels):
    model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=False)
    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value
    output_weights = tf.get_variable("output_weights", [num_labels, hidden_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
    
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, logits, probabilities)
def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps):
    def model_fn(features, labels, mode):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (total_loss, per_example_loss, logits, probabilities) = create_model(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                num_labels)
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
def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""
  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id)
  return feature
def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder,batch_size):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn():
    """The actual input function."""

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
def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    output_dir='./out'
    data_dir='./data'
    bert_config_file='./chinese_L-12_H-768_A-12/bert_config.json'
    vocab_file='./chinese_L-12_H-768_A-12/vocab.txt'
    do_lower_case=True
    log_steps=200
    do_train=True
    do_eval=False
    do_predict=True
    train_batch_size=8
    eval_batch_size=1
    test_batch_size=1
    num_train_epochs=2
    warmup_proportion=0.1
    init_checkpoint='./chinese_L-12_H-768_A-12/bert_model.ckpt'
    eval_checkpoint='./out/model.ckpt-3668'
    learning_rate=5e-5
    max_seq_length=128
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
    model_fn=model_fn_builder(bert_config, len(label_list), init_checkpoint, learning_rate,num_train_steps, num_warmup_steps)
    estimator=tf.estimator.Estimator(model_fn=model_fn,model_dir=output_dir,config=run_config)
    if do_train:
        train_file = os.path.join(output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
                train_examples, label_list, max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d",train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
                input_file=train_file,
                seq_length=max_seq_length,
                is_training=True,
                drop_remainder=True,
                batch_size=train_batch_size)
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
                batch_size=eval_batch_size)
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
    main()
