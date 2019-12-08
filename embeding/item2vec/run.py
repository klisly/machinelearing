#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-12-07 18:01
# @Author  : guifeng(moguifeng@baice100.com)
# @File    : run.py
import logging
import argparse
import time
import joblib
import numpy as np
import os
import shutil
import pandas as pd
from process import ItemNameProcessor
from embeding.item2vec import Options, Item2Vec
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

logging.basicConfig(level=logging.INFO)
logging.info("Start ...")
parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, help='Where did you put your data?',
                    required=True)

parser.add_argument('--save_path', type=str, help='The path you want to save your model.',
                    default='./result/item2vec/word/')

parser.add_argument('--save_project_path', type=str, help='The path you want to save your projected factors.',
                    default='./result/item2vec/item/')

parser.add_argument('--epochs', type=int, default=30,
                    help='Number of training epochs.')

parser.add_argument('--embedding_size', type=int, default=30,
                    help='The embedding dimension size.')

parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of examples per batch.')

parser.add_argument('--learning_rate', type=float, default=0.5,
                    help='Initial learning rate.')

parser.add_argument('--num_negatives', type=int, default=100,
                    help='Negative samples per training example.')
args = parser.parse_args()
start_time = time.time()
data = pd.read_csv(args.data)

processor = ItemNameProcessor(data, name_col='name')
opts = Options(args.embedding_size, args.batch_size,
               args.learning_rate, args.num_negatives, args.save_path)
with tf.Graph().as_default(), tf.Session() as session:
    with tf.device('/cpu:0'):
        model = Item2Vec(session, opts, processor)

    for epoch in range(args.epochs):
        model.train()
        model.evaluate('夏裝')
        if (epoch + 1) % 5 == 0:
            embeds = model.embeddings
            processor.print_similar(embeds, 10, N=10)

        print('Finish {} epoch!'.format(epoch + 1))
        print('-' * 10)
    embeds = model.embeddings
    processor.print_similar(embeds, 4839, N=10)
    processor.print_similar(embeds, 500, N=10)

    joblib.dump(embeds, os.path.join(args.save_path, 'word_embeds.pkl'))
    joblib.dump(processor, os.path.join(args.save_path, 'processor.pkl'))

with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
        if os.path.exists(args.save_project_path):
            shutil.rmtree(args.save_project_path, ignore_errors=False)
        if not os.path.exists(args.save_project_path):
            os.makedirs(args.save_project_path)
        f = processor.get_factors(embeds)
        f_tf = tf.convert_to_tensor(f, np.float64)

        factors = tf.Variable(f_tf, name='item_embedding')
        meta_path = os.path.join(args.save_project_path, 'item_metadata.tsv')
        meta_data = processor.get_item_meta()
        meta_data.to_csv(meta_path, sep='\t', index=False)
        saver = tf.train.Saver([factors])
        session.run(factors.initializer)
        saver.save(session, os.path.join(args.save_project_path, 'factors.ckpt'))
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = 'item_embedding'
        embedding.metadata_path = 'item_metadata.tsv'
        projector.visualize_embeddings(tf.summary.FileWriter(args.save_project_path), config)

logging.info('Complete in {} minutes!'.format((time.time() - start_time) / 60))
