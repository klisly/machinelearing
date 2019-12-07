#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-10-30 10:35
# @Author  : guifeng(moguifeng@baice100.com)
# @File    : linear.py

from mxnet.gluon import data as gdata
from mxnet import autograd, nd

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

batch_size = 10
dataset = gdata.ArrayDataset(features, labels)
data_iter = gdata.DataLoader(dataset, batch_size=batch_size, shuffle=True)

for X, y in data_iter:
    print(X, y)
    print("####")

# define model
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(1))

# init model param
from mxnet import init

net.initialize(init.Normal(0.1))

# define loss
from mxnet.gluon import loss as gloss

loss = gloss.L2Loss()

# define optimier
from mxnet import gluon

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

# train
num_epoche = 10
for epoch in range(num_epoche):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))

nn.MaxPool2D