#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/16 下午1:39
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : max_pool.py
import tensorflow as tf

a = tf.constant([1, 3, 2, 1, 2, 9, 1, 1, 1, 3, 2, 3, 5, 6, 1, 2, 5, 6, 1, 2], dtype=tf.float32, shape=[1, 4, 5, 1])
b = tf.nn.max_pool(a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
c = tf.nn.max_pool(a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
with tf.Session() as sess:
    print("a shape:")
    print(a.shape)
    print(sess.run(a))
    print("b shape:")
    print(b.shape)
    print("b value:")
    print(sess.run(b))
    print("c shape:")
    print(c.shape)
    print("c value:")
    print(sess.run(c))
