#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/16 下午1:37
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : relu.py
import tensorflow as tf
a = tf.constant([1, -2, 0, 4, -5,  6])
b = tf.nn.relu(a)
with tf.Session() as sess:
    print(sess.run(b))
    print(a.shape)