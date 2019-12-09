#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/18 下午11:58
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : dnn.py
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def add_layer(inputs, in_size, out_size, activation_function=None):
    W = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size]) + 0.01)

    Z = tf.matmul(inputs, W) + b
    if activation_function:
        outputs = activation_function(Z)
    else:
        outputs = Z
    return outputs


if __name__ == "__main__":
    MNIST = input_data.read_data_sets("./", one_hot=True)
    learning_rate = 0.01
    batch_size = 128
    n_epochs = 70

    X = tf.placeholder(tf.float32, [batch_size, 784])
    Y = tf.placeholder(tf.float32, [batch_size, 10])

    layer_dims = [784, 500, 500, 10]
    layer_count = len(layer_dims) - 1
    layer_iter = X

    for i in range(1, layer_count):
        layer_iter = add_layer(layer_iter, layer_dims[i-1], layer_dims[i], activation_function=tf.nn.relu)
    pred = add_layer(layer_iter, layer_dims[layer_count-1], layer_dims[layer_count], activation_function=None)
    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred)
    loss = tf.reduce_mean(entropy)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        n_batch = int(MNIST.train.num_examples / batch_size)
        for i in range(n_epochs):
            for j in range(n_batch):
                X_batch, Y_batch = MNIST.train.next_batch(batch_size)
                _, loss_ = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
                if j == 0:
                    print("Loss of epochs[{0}]: {1}".format(i, loss_))

        # test model
        n_batch = int(MNIST.test.num_examples / batch_size)
        totoal_correct_preds = 0
        for i in range(n_batch):
            X_batch, Y_batch = MNIST.test.next_batch(batch_size)
            preds = sess.run(pred, feed_dict={X: X_batch, Y: Y_batch})
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            totoal_correct_preds += sess.run(accuracy)
        print("Accuracy {0}".format(totoal_correct_preds / MNIST.test.num_examples))
