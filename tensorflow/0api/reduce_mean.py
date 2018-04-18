#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/15 上午9:33
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : reduce_mean.py
import tensorflow as tf
import numpy as np

initial = [[1., 1.], [2., 2.]]
x = tf.Variable(initial, dtype=tf.float32)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(tf.reduce_mean(x)))
    print(sess.run(tf.reduce_mean(x, 0)))
    print(sess.run(tf.reduce_mean(x, 1)))