#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/16 下午2:01
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : sigmoid_cross_entropy_with_logits.py
# 先对 logits 通过 sigmoid 计算，再计算交叉熵，交叉熵代价函数可以参考 CS231n: Convolutional Neural Networks for Visual Recognition

import tensorflow as tf
x = tf.constant([1,2,3,4,5,6,7],dtype=tf.float64)
y = tf.constant([1,1,1,0,0,1,0],dtype=tf.float64)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = y,logits = x)
with tf.Session() as sess:
    print (sess.run(loss))