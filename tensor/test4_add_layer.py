#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/19 上午12:57
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : test4_add_layer.py
import tensorflow as tf

def add_layer(inputs, in_size, out_size, activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size])) + 0.1
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs