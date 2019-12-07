#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/19 下午1:35
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : svr.py
# 支持分类的支持向量机可以推广到解决回归问题，这种方法称为支持向量回归
# 支持向量分类所产生的模型仅仅依赖于训练数据的一个子集，因为构建模型的成本函数不关心在超出边界范围的点，
# 类似的，通过支持向量回归产生的模型依赖于训练数据的一个子集，因为构建模型的函数忽略了靠近预测模型的数据集。
# 有三种不同的实现方式：支持向量回归SVR，nusvr和linearsvr。
# linearsvr提供了比SVR更快实施但只考虑线性核函数，而nusvr实现比SVR和linearsvr略有不同。
#
# 测试代码
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

y[::5] += 3 * (0.5 - np.random.rand(8))

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

lw = 2
plt.scatter(X, y, color='darkorange', label='data')
plt.hold('on')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()