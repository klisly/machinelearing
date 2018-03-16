#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/19 上午12:12
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : tensor2.py
import tensorflow as tf

state = tf.Variable(0, name='counter')
print(state.name)
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

#存在Variable变量就必须初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(3):
        sess.run(update)
        print(sess.run(state))
