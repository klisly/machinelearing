#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/16 下午1:54
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : dropout.py
import tensorflow as tf

a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3], dtype=tf.float32)
b = tf.placeholder(tf.float32)
c = tf.nn.dropout(a, b, [2, 3], 1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    print(sess.run(c, feed_dict={b: 0.75}))
