#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/2/7 下午3:05
# @Author  : guifeng(guifeng.mo@eyespage.cn)
# @File    : FeatureChoose.py
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
dataset =datasets.load_iris() # laod iris dataset
model = LogisticRegression() # build logistic regression model
# 作为一种特征选择方法，其工作原理是：循环地移除变量和建立模型，通过模型的准确率来评估变量对模型的贡献。
# 以下代码使用UCI的Iris数据集，使用sklearn.feature_selection的RFE方法来实现该方法。
rfe = RFE(model,2) # limit number of variables to three
rfe = rfe.fit(dataset.data,dataset.target)
print(rfe.support_)
print(rfe.ranking_)

from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier
dataset =datasets.load_iris() # laod iris dataset
model = ExtraTreesClassifier() # build extra tree model
model.fit(dataset.data,dataset.target)
print(model.feature_importances_) #display importance of each variables
