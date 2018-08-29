#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/29 上午11:00
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : bayes_test.py
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"
      % (iris.data.shape[0],(iris.target != y_pred).sum()))

from sklearn.ensemble import AdaBoostClassifier
