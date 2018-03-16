#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/3/16 下午2:03
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : truncated_normal.py
# 产生截断正态分布随机数，取值范围为 [ mean - 2 * stddev, mean + 2 * stddev ]。

import tensorflow as tf
initial = tf.truncated_normal(shape=[3,4], mean=0, stddev=1)
print(tf.Session().run(initial))