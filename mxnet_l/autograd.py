#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019-10-29 00:02
# @Author  : guifeng(moguifeng@baice100.com)
# @File    : mxnet.py
from mxnet import ndarray as nd
from mxnet import autograd

# x = nd.arange(4).reshape((4, 1))
# x.attach_grad()
# print(x)
# with autograd.record():
#     y = 2 * nd.dot(x.T, x)
# y.backwark()
# assert (x.grad - 4 * x).norm().asscalar() == 0
# print(x.grad)
# x = nd.array([[1, 2], [3, 4]])
# x.attach_grad()
# with autograd.record():
#     y = x * x
#     z = y * x * x
# z.backward()
# print(x.grad)

#导入autograd包
from mxnet import nd
from mxnet import autograd

#初始化输入函数的值
x = nd.array([1,2,3,4])
#利用attach_grad方法来保存梯度
x.attach_grad()
#定义前向传播的函数,保存函数便于计算梯度
with autograd.record():
    y =  x ** 3
#调用反向传播
y.backward()
#计算输入函数值的梯度
print(x.grad)