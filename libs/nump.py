#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/13 上午11:46
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : numpy.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

a = [1, 2, 3, 4]

b = np.array(a)
print(type(b))

print(b.shape)
print(b.argmax())
print(b.max())
print(b.mean())
