#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/22 7:11 PM
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : cnn_py.py
#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
from com.eyespage.textclassifier import data_helpers
from com.eyespage.textclassifier.text_cnn import TextCNN
from tensorflow.contrib import learn
from com.eyespage.comm.vertica_util import load_vertica_data
import traceback
# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .05, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# tf.flags.DEFINE_string("positive_data_file", "/Users/john/eclipse/workspace/ExpiPy/data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "/Users/john/eclipse/workspace/ExpiPy/data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 24, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 4, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 16, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")


cdict = {}

r_cidx = []
r_names = []

def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
#     x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    baike = load_vertica_data(
    #     "select * from john_tagged_baike_02_sample_vector where random() < 1000/co "
    """
    select
        decode(idx, 22,2,23,3,24,11,idx) as idx
        , co, domain, content from dev.fortrain_keyword_search;
    """
#         """
#         select
#             *
#         from
#             john_tagged_baidubaike_content03 where random() < 0.1
#         ;
#         """

#         """
#         select * from (
#             select 1 as idx, 221566 as co, 'person' as domain, regexp_replace(content, '(?<=.)(?=.)', ' ') as content
#             from john_baike_manually_tag_corpus_01 where id ~ '^[李王张刘陈杨黄吴周赵孙朱徐马林胡郑郭何][一-龥]{1,2}/' or tag = '人物'
#             union all
#             select 2 as idx, 958138 as co, 'other' as domain, regexp_replace(content, '(?<=.)(?=.)', ' ') as content
#             from john_baike_manually_tag_corpus_01 where not (id ~ '^[^李王张刘陈杨黄吴周赵孙朱徐马林胡郑郭何][一-龥]{1,2}/' or tag = '人物')
#         ) as a where random() < 0.1
#         ;
#         """

#         """
#         select * from (
#             select 1 as idx, 22 as co, 'person' as domain, regexp_replace(regexp_replace(id, '/[0-9]+$', ''), '(?<=.)(?=.)', ' ') as content
#             from john_baike_person_predict_step01 where id ~ '[一-龥]{2,}' and predicted = 'person' and not ( id ~ '.{3,}/' and regexp_replace(id,'/[0-9]+$','') ~ '[^一-龥·]')
#             union all
#             select 2 as idx, 33 as co, 'other' as domain, regexp_replace(regexp_replace(id, '/[0-9]+$', ''), '(?<=.)(?=.)', ' ') as content
#             from john_baike_person_predict_step01 where not ( id ~ '[一-龥]{2,}' and predicted = 'person' and not ( id ~ '.{3,}/' and regexp_replace(id,'/[0-9]+$','') ~ '[^一-龥·]'))
#         ) as a where random() < 0.1
#         ;
#         """


#     select * from (select idx, co, domain, regexp_substr(c,'(?<=/wiki/baidu/).+(?=_)') as content from john_tagged_baidubaike_content02 as a
#         where exists (select * from john_baike_attr_co where c = a.c and count>5)) as a where random() < 300/co
#     """
#     "select idx, co, domain, content from john_tagged_baidubaike_content02 where random() < 200/co and (select * from john_baike_attr_co where c = a.c and count>10) -- or (domain in('car') and random()< 1500/co) "
#     "select decode(domain, 'car', 1, 2) as idx, co, domain, content from john_tagged_baidubaike_content02 where random() < 100/co or (domain in('car') and random()< 1500/co) "
#     """
#     select 1 as idx, co, 'car' as domain, content from john_tagged_baidubaike_content02 where c ~ '/刘' and random()<1
#     union
#     select 2 as idx, co, 'other' as domain, content from john_tagged_baidubaike_content02 where random() < 0.0001
#     """
#     "select * from john_baike_vector_sample_train"


#     , host = '172.31.2.140'
#     , database='eyespage_data_platform'

        , host='10.0.0.23'
        , user='dba'
        , database='gsearch'
    )

    x_text = []
    y = []

    for d in baike:
        try:
            x_text.append(d[3])
            a = [0] * 21
            a[d[0]-1] = 1
            cdict[d[0]-1] = d[2]
            y.append(a)
        except :
            print(d)
            traceback.print_exc()
            pass


    for i, key in enumerate(cdict):
        r_cidx.append(key)
        r_names.append(cdict[key])

    y = np.array(y)
#     print(y)
#     print(x_text)

    # Build vocabulary
    #max_document_length = max([len(x.split(" ")) for x in x_text])
    #vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore('person_baike_cnn_vocab_processor.pkl')
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    #vocab_processor.save('person_baike_cnn_vocab_processor.pkl')

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
#     print("shuffle_indices", shuffle_indices)
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev


from sklearn.metrics import classification_report

def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
#                 print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy, y_pred, test_y = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions, cnn.test_y],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()

                print(test_y)
                print(y_pred)
                print(classification_report(test_y, y_pred, labels=r_cidx, target_names=r_names))
#                 classification_report.

                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()