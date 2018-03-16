#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/28 上午11:40
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : test.py
from __future__ import print_function
import torch
from torch.autograd import Variable

x = Variable(torch.ones(2,2), requires_grad=True)
print (x)

y = x + 2
print(y)