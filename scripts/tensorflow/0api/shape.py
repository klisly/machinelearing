  #!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/15 上午9:06
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : shape.py
import tensorflow as tf

tensor=tf.constant([1,2,3,4,5,6,7,8,9])
print(tensor.shape)
print(tf.shape(tensor))
print(tensor.get_shape())
tensor=tf.constant([[1,2,3],[4,5,6],[7,8,9]])
print(tensor.shape)
print(tf.shape(tensor))
print(tensor.get_shape())
tensor=tf.constant([[[1,1,1],[2,2,3],[3,3,3]],[[4,4,4],[5,5,5],[6,6,6]]])
print(tensor.shape)
print(tf.shape(tensor))
print(tensor.get_shape())
t = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
tf.shape(t)  # [2, 2, 3]

