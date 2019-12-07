#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-12-07 00:43
# @Author  : guifeng(moguifeng@baice100.com)
# @File    : item2vec.py
import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from itertools import combinations
from collections import deque


class Options(object):
    def __init__(self, embedding_size, batch_size,
                 learning_rate, num_negatives, save_path):
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_negatives = num_negatives
        self.save_path = save_path


class BatchGenerator(object):
    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.ix = 0
        self.buffer = deque([])
        self.is_finish = False

    def next(self):
        if self.is_finish:
            return None

        while len(self.buffer) < self.batch_size:
            items_list = self.data.iloc[self.ix]
            self.buffer.extend(combinations(items_list, 2))
            if self.ix == self.data.shape[0] - 1:
                self.is_finish = True
                self.ix = 0
            else:
                self.ix += 1
        d = [self.buffer.popleft() for _ in range(self.batch_size)]
        d = np.array([[i[0], i[1]] for i in d])
        batch = d[:, 0]
        label = d[:, 1]
        return batch, label

    @property
    def finish(self):
        return self.is_finish

    def resume(self):
        self.ix = 0
        self.is_finish = False

    @property
    def current_percentage(self):
        return (self.ix / self.data.shape[0]) * 100


class Item2Vec(object):

    def __init__(self, session, opts, processor):
        self.vocab_size = len(processor.word_list)
        self.embed_dim = opts.embedding_size
        self.num_negatives = opts.num_negatives
        self.learning_rate = opts.learning_rate
        self.batch_size = opts.batch_size
        self.save_path = opts.save_path
        self.step = 0

        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path, ignore_errors=True)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.item_counts = processor.word_counts
        self.generator = BatchGenerator(opts.batch_size, processor.clean_data)

        self.processor = processor
        meta_path = os.path.join(self.save_path, 'word_metadata.tsv')
        meta_data = self.processor.get_word_meta()
        meta_data.to_csv(meta_path, sep='\t', index=False)

        self.projector_config = projector.ProjectorConfig()
        pro_embed = self.projector_config.embeddings.add()
        pro_embed.tensor_name = 'word_embedding'
        pro_embed.metadata_path = 'word_metadata.tsv'
        self.session = session
        self._init_graph()

    def _init_graph(self):
        self.batch = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        true_logits, sampled_logits = self.forward(self.batch, self.labels)
        self.loss = self.nce_loss(true_logits, sampled_logits)
        self.train_op = self.optimize(self.loss)
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.save_path, graph=tf.get_default_graph())
        projector.visualize_embeddings(self.summary_writer, self.projector_config)

    def forward(self, batch, labels):
        init_width = 0.5 / self.embed_dim
        embed = tf.Variable(tf.random_normal([self.vocab_size, self.embed_dim],
                                             -init_width, init_width), name='word_embedding')
        self.embed = embed
        softmax_w = tf.Variable(tf.zeros([self.vocab_size, self.embed_dim]), name='softmax_weights')
        softmax_b = tf.Variable(tf.zeros([self.vocab_size]), name='softmax_bias')
        labels_matrix = tf.reshape(tf.cast(labels, tf.int64), [self.batch_size, 1])
        sample_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels_matrix, num_true=1, num_sampled=self.num_negatives, unique=True,
            range_max=self.vocab_size, distortion=0.75, unigrams=self.item_counts)

        example_emb = tf.nn.embedding_lookup(embed, batch)
        true_w = tf.nn.embedding_lookup(softmax_w, labels)
        true_b = tf.nn.embedding_lookup(softmax_b, labels)
        true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w) + 1) + true_b

        sample_w = tf.nn.embedding_lookup(softmax_w, sample_ids)
        sample_b = tf.nn.embedding_lookup(softmax_b, sample_ids)
        sample_b_vec = tf.reshape(sample_b, [self.num_negatives])
        sample_logits = tf.matmul(example_emb, sample_w, transpose_b=True) + sample_b_vec

        return true_logits, sample_logits

    def nce_loss(self, true_logits, sampled_logits):
        true_x = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(true_logits), logits=true_logits)
        sample_x = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(sampled_logits), logits=sampled_logits)
        nce_loss = (tf.reduce_sum(true_x) + tf.reduce_sum(sample_x)) / self.batch_size
        return nce_loss

    def optimize(self, loss):
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        train_op = opt.minimize(loss)
        tf.summary.scalar("loss", loss)
        self.summary_op = tf.summary.merge_all()
        return train_op

    def train(self):
        avg_loss = 0
        while not self.generator.finish:
            batch, labels = self.generator.next()
            feed_dict = {self.batch: batch, self.labels: labels}
            _, loss_val, summary = self.session.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, self.step)
            avg_loss += loss_val
            self.step += 1
        print("Cost: ", '{:.9f}'.format(avg_loss))
        self.generator.resume()
        self.saver.save(self.session, os.path.join(self.save_path, "model.ckpt"), global_step=self.step)

    @property
    def embeddings(self):
        return self.embed.eval()

    def get_norms(self):
        norms = np.linalg.norm(self.embeddings, axis=-1)
        norms[norms == 0] = 1e-10
        return norms

    def similar_items(self, itemid, N=10):
        norms = self.get_norms()
        scores = self.embeddings.dot(self.embeddings[itemid]) / norms
        best = np.argpartition(scores, -N)[-N:]
        return sorted(zip(best, scores[best] / norms[itemid]), key=lambda x: -x[1])

    def evaluate(self, word):
        idx = self.processor.word_to_idx[word]
        for i, score in self.similar_items(idx):
            print(self.processor.word_list[i], score)
        print('-' * 10)
