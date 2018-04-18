#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/16 下午2:05
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : placeholder.py
# 是一种占位符，在执行时候需要为其提供数据

import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float16, [None, 3])
y = tf.matmul(x, x)
with tf.Session() as sess:
    rnd = np.random.rand(3, 3)
    print(sess.run(y, feed_dict={x: rnd}))
