#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/19 上午12:12
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : tensor2.py
import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input2:23, input1:23}))
