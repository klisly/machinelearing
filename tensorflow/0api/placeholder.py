#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/16 下午2:05
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : placeholder.py
# 是一种占位符，在执行时候需要为其提供数据

import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32,[4,4])
y = tf.matmul(x,x)
with tf.Session() as sess:
    rand_array = np.random.rand(4,4)
    print(sess.run(y,feed_dict={x:rand_array}))