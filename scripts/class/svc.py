#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/19 下午1:34
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : svc.py
# 1、不平衡数据分类问题
#
# 对于非平衡级分类超平面，使用不平衡SVC找出最优分类超平面，基本的思想是，我们先找到一个普通的分类超平面，自动进行校正，求出最优的分类超平面
#
# 测试代码如下：

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

rng = np.random.RandomState(0)
n_samples_1 = 1000
n_samples_2 = 100
X = np.r_[1.5 * rng.randn(n_samples_1, 2), 0.5 * rng.randn(n_samples_2, 2) + [2, 2]]
y = [0] * (n_samples_1) + [1] * (n_samples_2)
print(X)
print(y)

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - clf.intercept_[0] / w[1]
wclf = svm.SVC(kernel='linear', class_weight={1: 10})
wclf.fit(X, y)

ww = wclf.coef_[0]
wa = -ww[0] / ww[1]
wyy = wa * xx - wclf.intercept_[0] / ww[1]

h0 = plt.plot(xx, yy, 'k-', label='no weights')
h1 = plt.plot(xx, wyy, 'k--', label='with weights')
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.legend()

plt.axis('tight')
plt.show()
