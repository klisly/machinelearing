#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/28 上午11:40
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : test.py
from __future__ import print_function
import torch

x = torch.Tensor(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)

print(x.size())

y = torch.rand(5, 3)
print(y)
print(x + y)

result = torch.Tensor(5, 3)
torch.add(x, y, out=result)
print(result)