#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/18 上午9:54
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : test1.py
import tensorflow as tf
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)
print("x_data:",x_data)
y_data = x_data * 0.1 + 0.3
print("y_data",y_data)

#create tensorflow structure start
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

print(Weights)
print(biases)

y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
# create tensorflow structure end
sess = tf.Session()
sess.run(init)

for step in range(1000):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
print(step, sess.run(Weights), sess.run(biases))