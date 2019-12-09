#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/16 下午2:05
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : constant.py
# 根据 value 的值生成一个 shape 维度的常量张量


import tensorflow as tf
import numpy as np

a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b = tf.constant(-1, shape=[3, 2])
c = tf.matmul(a, b)

e = tf.constant(np.arange(1, 13, dtype=np.int32), shape=[2, 2, 3])
f = tf.constant(np.arange(13, 25, dtype=np.int32), shape=[2, 3, 2])
g = tf.matmul(e, f)
with tf.Session() as sess:
    print(sess.run(a))
    print("##################################")
    print(sess.run(b))
    print("##################################")
    print(sess.run(c))
    print("##################################")
    print(sess.run(e))
    print("##################################")
    print(sess.run(f))
    print("##################################")
    print(sess.run(g))
